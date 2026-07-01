`timescale 1ns / 1ps
// ============================================================================
// global_avg_pool
//   Global average pooling: HxWxC input -> C output values (one spatial
//   average per channel). Parameterized on IMG_W/IMG_H/IN_CH for reuse
//   across different feature map sizes.
//
//   Memory layout convention (matches conv_layer/maxpool exactly, so this
//   module can read conv_layer's or maxpool's ofmap memory directly):
//     ifmap : addr = (ic * IMG_H + iy) * IMG_W + ix   (channel-major plane)
//     ofmap : addr = ic                                (no spatial dims - one
//                    value per channel, so the address IS the channel index)
//
//   Output width: ofmap_wdata is 'signed [DATA_W-1:0]', DATA_W=8 by default
//   - the exact same declaration as conv_layer's ofmap_wdata and maxpool's
//   ifmap_rdata/ofmap_wdata. No width unification trick is needed here
//   beyond simply using the same DATA_W/port declaration style; there is
//   no separate "16-bit intermediate" port anywhere in this module.
//
//   Fixed-point handling (no FRAC_BITS/shift parameter, unlike conv_layer):
//   averaging never changes the fixed-point scale - summing N numbers that
//   all share the same activation frac_bits (a system-wide constant fixed
//   in task05_export_weights.py, see quant_meta.json) and dividing by N
//   yields a result at that same frac_bits. There is no multiply here
//   (unlike conv's activation*weight, which produces a wider product
//   needing a right-shift back down to the activation's frac_bits).
//   So the only fixed-point operation is a plain integer divide by
//   N_PIX = IMG_W*IMG_H, which is a compile-time constant (not a power of
//   two in general - e.g. 6x6=36) so it's implemented as an actual integer
//   division, not a shift.
//
//   Rounding: round-half-away-from-zero, computed via sign/magnitude
//   handling instead of relying on Verilog's native '/' (which truncates
//   toward zero for both signed operands, i.e. plain truncation, not
//   rounding, and truncation is directionally biased for negative sums).
//     acc >= 0 :  avg = (acc + N_PIX/2) / N_PIX
//     acc <  0 :  avg = -((-acc + N_PIX/2) / N_PIX)
//   i.e. round the magnitude of acc/N_PIX to the nearest integer (ties round
//   up in magnitude), then re-apply the sign. This gives symmetric rounding
//   for negative sums instead of Verilog's default toward-zero truncation
//   bias. No saturation is applied on the result: the average of values
//   that are each within [-2^(DATA_W-1), 2^(DATA_W-1)-1] can mathematically
//   never fall outside that same range, so truncating avg_val to the low
//   DATA_W bits is always exact.
//
//   Same BRAM timing assumption and FETCH/CAP/WRITE state-per-cycle style
//   as conv_layer/maxpool: address held stable for one full cycle ->
//   registered read data valid the following cycle.
// ============================================================================
module global_avg_pool #(
    parameter IMG_W  = 6,
    parameter IMG_H  = 6,
    parameter IN_CH  = 64,
    parameter DATA_W = 8,   // activation width - matches conv_layer/maxpool (frac_bits is external context)
    parameter ACC_W  = 32,  // internal sum accumulator width
    parameter ADDR_W = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  busy,
    output reg  done,

    // input feature map (read-only) - pre-pool size (IMG_W x IMG_H x IN_CH)
    output reg  [ADDR_W-1:0]        ifmap_addr,
    input  wire signed [DATA_W-1:0] ifmap_rdata,

    // output: one averaged value per channel (write-only)
    output reg                      ofmap_we,
    output reg  [ADDR_W-1:0]        ofmap_addr,
    output reg  signed [DATA_W-1:0] ofmap_wdata
);

    localparam signed [31:0] N_PIX = IMG_W * IMG_H;

    // ------------------------------------------------------------------
    // address helper function (pure, no state)
    // ------------------------------------------------------------------
    function integer ifmap_index;
        input integer ic_f, iy_f, ix_f;
        begin
            ifmap_index = (ic_f * IMG_H + iy_f) * IMG_W + ix_f;
        end
    endfunction

    // ------------------------------------------------------------------
    // loop counters - plain 'integer' (32-bit signed), same rationale as
    // conv_layer/maxpool.
    // ------------------------------------------------------------------
    integer ic, py, px;
    integer next_py, next_px;

    wire pix_is_last   = (px == IMG_W-1) && (py == IMG_H-1);
    wire layer_is_last = (ic == IN_CH-1);

    localparam S_IDLE  = 3'd0,
               S_FETCH = 3'd1,  // ifmap_addr held stable this cycle
               S_CAP   = 3'd2,  // ifmap_rdata now valid; add into running sum, issue next address
               S_WRITE = 3'd3,  // divide+round, pulse ofmap_we, advance to next channel
               S_DONE  = 3'd4;

    reg [2:0] state;
    reg signed [ACC_W-1:0] acc;
    reg signed [ACC_W-1:0] abs_acc, avg_val; // S_WRITE scratch (blocking-assigned, same pattern as conv_layer's relu_val/shifted_val/sat_val)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            ofmap_we    <= 1'b0;
            ic <= 0; py <= 0; px <= 0;
            acc         <= {ACC_W{1'b0}};
            ifmap_addr  <= {ADDR_W{1'b0}};
            ofmap_addr  <= {ADDR_W{1'b0}};
            ofmap_wdata <= {DATA_W{1'b0}};
        end else begin
            ofmap_we <= 1'b0;
            done     <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy       <= 1'b1;
                    ic <= 0; py <= 0; px <= 0;
                    acc        <= {ACC_W{1'b0}};
                    ifmap_addr <= ifmap_index(0, 0, 0);
                    state      <= S_FETCH;
                end
            end

            // ----------------------------------------------------------
            S_FETCH: begin
                state <= S_CAP;
            end

            // ----------------------------------------------------------
            S_CAP: begin
                // ifmap_rdata now corresponds to the address issued during
                // S_FETCH (== current ic/py/px). Unlike maxpool's running
                // max, sum has no special "first tap" case: starting acc
                // at 0 and always adding is correct for every tap.
                acc <= acc + $signed(ifmap_rdata);

                if (pix_is_last) begin
                    state <= S_WRITE;
                end else begin
                    if (px == IMG_W-1) begin
                        next_px = 0;
                        next_py = py + 1;
                    end else begin
                        next_px = px + 1;
                        next_py = py;
                    end
                    px <= next_px;
                    py <= next_py;
                    ifmap_addr <= ifmap_index(ic, next_py, next_px);
                    state <= S_FETCH;
                end
            end

            // ----------------------------------------------------------
            S_WRITE: begin
                // round-half-away-from-zero divide by N_PIX (see header comment)
                if (acc >= 0) begin
                    avg_val = (acc + (N_PIX/2)) / N_PIX;
                end else begin
                    abs_acc = -acc;
                    avg_val = -((abs_acc + (N_PIX/2)) / N_PIX);
                end

                ofmap_we    <= 1'b1;
                ofmap_addr  <= ic;                  // no spatial dims: address is just the channel index
                ofmap_wdata <= avg_val[DATA_W-1:0]; // exact: avg of DATA_W-range inputs is always within DATA_W range

                py <= 0; px <= 0;
                acc <= {ACC_W{1'b0}};

                if (!layer_is_last) begin
                    ic         <= ic + 1;
                    ifmap_addr <= ifmap_index(ic+1, 0, 0);
                    state      <= S_FETCH;
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
