`timescale 1ns / 1ps
// ============================================================================
// conv_layer
//   3x3 "same"-padding convolution + bias + ReLU. Parameterized so the same
//   RTL is reused for conv1/conv2/conv3 by instantiating with different
//   IMG_W/IMG_H/IN_CH/OUT_CH. Spatial size in == spatial size out (pooling
//   is a separate module, applied after this one).
//
//   Fixed-point convention: this module is agnostic to what its integers
//   "mean" as real numbers - it only ever does raw int8/int16 arithmetic on
//   DATA_W/BIAS_W-bit values, so the specific frac_bits interpretation
//   (chosen in task05_export_weights.py, see quant_meta.json) lives entirely
//   outside this file. The convention as currently used system-wide:
//     activation : DATA_W=8 bits, frac=ACT_FRAC_BITS (a system-wide
//                  constant chosen in the export script - was 7 (Q1.7)
//                  originally but raised to frac=2 after intermediate
//                  activations were found to regularly exceed the Q1.7
//                  range of 0..0.992, causing a class-collapse bug; see
//                  tb_cnn_core_fsm.v for the full writeup)
//     weight     : DATA_W=8 bits, frac=FRAC_BITS (this instance's
//                  parameter - chosen per-layer, NOT assumed to be 7)
//     bias       : BIAS_W=16 bits, frac = ACT_FRAC_BITS + FRAC_BITS,
//                  matching the scale of an activation*weight product so it
//                  can be added directly into the accumulator
//     accumulator: ACC_W-bit, same frac as bias above
//   After all taps are summed: ReLU (clip negative to 0) -> right shift by
//   FRAC_BITS (back to the activation's frac_bits) -> saturate to
//   [0, 2^(DATA_W-1)-1] -> ofmap.
//
//   Memory layout convention (the caller / external BRAMs must match this):
//     ifmap : addr = (ic * IMG_H + iy) * IMG_W + ix      (channel-major plane)
//     weight: addr = ((oc * IN_CH + ic) * KSIZE + ky) * KSIZE + kx
//     bias  : addr = oc
//     ofmap : addr = (oc * IMG_H + oy) * IMG_W + ox      (channel-major plane)
//   This matches task05_export_weights.py's flatten order: conv weights are
//   flattened as (out_ch, in_ch, kh, kw).
//
//   All four memories are assumed to be external single-port synchronous
//   BRAMs with standard registered-read behavior: address held stable for
//   one full clock cycle -> read data valid the following cycle. This
//   module never assumes 0-cycle (combinational) read data, and never
//   changes an address register mid-cycle.
// ============================================================================
module conv_layer #(
    parameter IMG_W     = 48,
    parameter IMG_H     = 48,
    parameter IN_CH     = 1,
    parameter OUT_CH    = 8,
    parameter KSIZE     = 3,
    parameter DATA_W    = 8,    // activation / weight width (frac_bits is external context, not fixed here)
    parameter BIAS_W    = 16,   // Q2.14 bias width
    parameter ACC_W     = 32,   // internal accumulator width
    parameter FRAC_BITS = 7,    // right-shift applied after bias-add, before ReLU/saturate
    parameter ADDR_W    = 16    // generous fixed address width for all 4 memory ports
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  busy,
    output reg  done,

    // input feature map (read-only)
    output reg  [ADDR_W-1:0]        ifmap_addr,
    input  wire signed [DATA_W-1:0] ifmap_rdata,

    // weight memory (read-only)
    output reg  [ADDR_W-1:0]        weight_addr,
    input  wire signed [DATA_W-1:0] weight_rdata,

    // bias memory (read-only)
    output reg  [ADDR_W-1:0]        bias_addr,
    input  wire signed [BIAS_W-1:0] bias_rdata,

    // output feature map (write-only)
    output reg                      ofmap_we,
    output reg  [ADDR_W-1:0]        ofmap_addr,
    output reg  signed [DATA_W-1:0] ofmap_wdata
);

    localparam PAD = (KSIZE - 1) / 2;
    localparam signed [ACC_W-1:0] SAT_MAX = (1 << (DATA_W-1)) - 1; // 127 for DATA_W=8

    // ------------------------------------------------------------------
    // address helper functions (pure, no state) - kept as functions so the
    // index formula is written exactly once and used identically for every
    // term, instead of being re-derived ad hoc at each call site.
    // ------------------------------------------------------------------
    function integer ifmap_index;
        input integer ic_f, iy_f, ix_f;
        begin
            ifmap_index = (ic_f * IMG_H + iy_f) * IMG_W + ix_f;
        end
    endfunction

    function integer weight_index;
        input integer oc_f, ic_f, ky_f, kx_f;
        begin
            weight_index = ((oc_f * IN_CH + ic_f) * KSIZE + ky_f) * KSIZE + kx_f;
        end
    endfunction

    function integer ofmap_index;
        input integer oc_f, oy_f, ox_f;
        begin
            ofmap_index = (oc_f * IMG_H + oy_f) * IMG_W + ox_f;
        end
    endfunction

    // ------------------------------------------------------------------
    // loop counters - plain 'integer' (32-bit signed) on purpose: avoids
    // unsigned-wrap bugs when computing "oy + ky - PAD", which is
    // transiently negative at the top/left border before the padding
    // check clamps it.
    // ------------------------------------------------------------------
    integer oc, oy, ox, ic, ky, kx;

    // scratch variables for "next term" address computation (blocking-
    // assigned, recomputed fresh every time before use - never relied on
    // to hold a value across cycles)
    integer next_ic, next_ky, next_kx, next_iy, next_ix;
    reg signed [ACC_W-1:0] relu_val, shifted_val, sat_val;

    wire term_is_last  = (ic == IN_CH-1)  && (ky == KSIZE-1) && (kx == KSIZE-1);
    wire pixel_is_last = (ox == IMG_W-1)  && (oy == IMG_H-1);
    wire layer_is_last = (oc == OUT_CH-1);

    localparam S_IDLE      = 3'd0,
               S_BIAS_ADDR = 3'd1,   // bias_addr held stable this cycle (set by previous state)
               S_BIAS_CAP  = 3'd2,   // bias_rdata now valid; also issues address for the first MAC term
               S_FETCH     = 3'd3,   // ifmap_addr/weight_addr held stable this cycle
               S_COMPUTE   = 3'd4,   // ifmap_rdata/weight_rdata now valid; do the MAC, issue next address
               S_WRITE     = 3'd5,   // apply ReLU/shift/saturate, pulse ofmap_we, advance to next pixel/oc
               S_DONE      = 3'd6;

    reg [2:0] state;
    reg signed [BIAS_W-1:0] bias_reg;
    reg signed [ACC_W-1:0]  acc;
    reg                     pad_reg;  // registered "this tap is zero-padding" flag: set in the state that issues the address, consumed one state later when data arrives

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            ofmap_we    <= 1'b0;
            oc <= 0; oy <= 0; ox <= 0; ic <= 0; ky <= 0; kx <= 0;
            acc         <= {ACC_W{1'b0}};
            bias_reg    <= {BIAS_W{1'b0}};
            pad_reg     <= 1'b0;
            ifmap_addr  <= {ADDR_W{1'b0}};
            weight_addr <= {ADDR_W{1'b0}};
            bias_addr   <= {ADDR_W{1'b0}};
            ofmap_addr  <= {ADDR_W{1'b0}};
            ofmap_wdata <= {DATA_W{1'b0}};
        end else begin
            // defaults: ofmap_we/done are 1-cycle pulses unless a branch below re-asserts them
            ofmap_we <= 1'b0;
            done     <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy      <= 1'b1;
                    oc <= 0; oy <= 0; ox <= 0; ic <= 0; ky <= 0; kx <= 0;
                    bias_addr <= 0;
                    state     <= S_BIAS_ADDR;
                end
            end

            // ----------------------------------------------------------
            S_BIAS_ADDR: begin
                // bias_addr was already set (by S_IDLE above, or by S_WRITE
                // below when moving to the next oc) at the edge entering
                // this state, so it's stable for this whole cycle.
                state <= S_BIAS_CAP;
            end

            // ----------------------------------------------------------
            S_BIAS_CAP: begin
                bias_reg <= bias_rdata;
                acc      <= {{(ACC_W-BIAS_W){bias_rdata[BIAS_W-1]}}, bias_rdata}; // sign-extend Q2.14 bias into the accumulator

                // issue the address for the very first MAC term (ic=ky=kx=0)
                // of this pixel; oy/ox are already correct, ic/ky/kx are 0
                next_iy = oy - PAD;
                next_ix = ox - PAD;
                if (next_iy < 0 || next_iy >= IMG_H || next_ix < 0 || next_ix >= IMG_W) begin
                    pad_reg    <= 1'b1;
                    ifmap_addr <= ifmap_index(0, 0, 0); // dummy address; value unused due to pad_reg
                end else begin
                    pad_reg    <= 1'b0;
                    ifmap_addr <= ifmap_index(0, next_iy, next_ix);
                end
                weight_addr <= weight_index(oc, 0, 0, 0);
                state       <= S_FETCH;
            end

            // ----------------------------------------------------------
            S_FETCH: begin
                // address set last cycle is now presented to the BRAMs;
                // wait one cycle for synchronous read data to appear
                state <= S_COMPUTE;
            end

            // ----------------------------------------------------------
            S_COMPUTE: begin
                // ifmap_rdata / weight_rdata now correspond to the address
                // issued during S_FETCH (== current ic/ky/kx/oy/ox/oc)
                if (!pad_reg)
                    acc <= acc + ($signed(ifmap_rdata) * $signed(weight_rdata));

                if (term_is_last) begin
                    state <= S_WRITE;
                end else begin
                    // compute (ic,ky,kx) of the NEXT term
                    if (kx == KSIZE-1) begin
                        next_kx = 0;
                        if (ky == KSIZE-1) begin
                            next_ky = 0;
                            next_ic = ic + 1;
                        end else begin
                            next_ky = ky + 1;
                            next_ic = ic;
                        end
                    end else begin
                        next_kx = kx + 1;
                        next_ky = ky;
                        next_ic = ic;
                    end

                    ic <= next_ic;
                    ky <= next_ky;
                    kx <= next_kx;

                    next_iy = oy + next_ky - PAD;
                    next_ix = ox + next_kx - PAD;

                    if (next_iy < 0 || next_iy >= IMG_H || next_ix < 0 || next_ix >= IMG_W) begin
                        pad_reg    <= 1'b1;
                        ifmap_addr <= ifmap_index(next_ic, 0, 0); // dummy
                    end else begin
                        pad_reg    <= 1'b0;
                        ifmap_addr <= ifmap_index(next_ic, next_iy, next_ix);
                    end
                    weight_addr <= weight_index(oc, next_ic, next_ky, next_kx);

                    state <= S_FETCH;
                end
            end

            // ----------------------------------------------------------
            S_WRITE: begin
                relu_val    = acc[ACC_W-1] ? {ACC_W{1'b0}} : acc;      // ReLU
                shifted_val = relu_val >>> FRAC_BITS;                  // back to activation frac_bits (value >=0, arithmetic==logical shift)
                sat_val     = (shifted_val > SAT_MAX) ? SAT_MAX : shifted_val;

                ofmap_we    <= 1'b1;
                ofmap_addr  <= ofmap_index(oc, oy, ox);
                ofmap_wdata <= sat_val[DATA_W-1:0];

                ic <= 0; ky <= 0; kx <= 0;

                if (!pixel_is_last) begin
                    // same oc -> reuse bias_reg, no need to re-read bias memory
                    if (ox == IMG_W-1) begin
                        ox <= 0;
                        oy <= oy + 1;
                        next_iy = (oy + 1) - PAD;
                        next_ix = 0 - PAD;
                    end else begin
                        ox <= ox + 1;
                        next_iy = oy - PAD;
                        next_ix = (ox + 1) - PAD;
                    end

                    if (next_iy < 0 || next_iy >= IMG_H || next_ix < 0 || next_ix >= IMG_W) begin
                        pad_reg    <= 1'b1;
                        ifmap_addr <= ifmap_index(0, 0, 0);
                    end else begin
                        pad_reg    <= 1'b0;
                        ifmap_addr <= ifmap_index(0, next_iy, next_ix);
                    end
                    weight_addr <= weight_index(oc, 0, 0, 0);
                    acc         <= {{(ACC_W-BIAS_W){bias_reg[BIAS_W-1]}}, bias_reg};
                    state       <= S_FETCH;

                end else if (!layer_is_last) begin
                    // move to next output channel -> must reload bias
                    oy        <= 0;
                    ox        <= 0;
                    oc        <= oc + 1;
                    bias_addr <= oc + 1;
                    state     <= S_BIAS_ADDR;

                end else begin
                    state <= S_DONE;
                end
            end

            // ----------------------------------------------------------
            S_DONE: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
