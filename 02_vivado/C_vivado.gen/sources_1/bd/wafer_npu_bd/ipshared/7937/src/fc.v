`timescale 1ns / 1ps
// ============================================================================
// fc (dense / fully-connected)
//   IN_FEATURES -> OUT_FEATURES fully-connected layer. This is the final
//   layer of the model (GAP -> fc -> argmax), so NO ReLU is applied here -
//   the raw (bias-added, shifted) class scores are written out as-is for
//   argmax to compare, matching the spec ("마지막 레이어라 ReLU는 적용하지
//   않음 - argmax로 바로 넘어감").
//
//   Fixed-point convention (same shape as conv_layer - this module is
//   agnostic to what its integers "mean" as real numbers, see conv_layer.v):
//     activation : DATA_W=8 bits, frac=ACT_FRAC_BITS - a system-wide
//                  constant fixed in task05_export_weights.py (currently 2,
//                  see quant_meta.json), since fc's input must match
//                  global_avg_pool's output scale exactly.
//     weight     : DATA_W=8 bits, frac=FRAC_BITS' where FRAC_BITS' is THIS
//                  instance's FRAC_BITS parameter - NOT assumed to be equal
//                  to ACT_FRAC_BITS. task05_export_weights.py picks
//                  FRAC_BITS per layer to avoid clipping; the fc layer's
//                  trained weights have |w|max=9.14, far outside what a
//                  small frac_bits could represent without heavy clipping,
//                  so the real model's fc instance is quantized with
//                  FRAC_BITS=3 (range +-15.875).
//     bias       : BIAS_W=16 bits, frac = ACT_FRAC_BITS + FRAC_BITS' (this
//                  layer's weight), matching the accumulator's scale at the
//                  point of addition.
//     accumulator: ACC_W-bit, same frac as bias above.
//   acc = bias + sum(ifmap*weight), then a plain arithmetic right-shift by
//   FRAC_BITS. No ReLU, no saturation (see below for why saturation is
//   unnecessary here). Because FRAC_BITS varies per instantiation, the
//   REQUIRED SCORE_W varies too - see the margin analysis below, which is
//   written as a formula for exactly this reason, not a fixed number.
//
//   ---- Weight indexing: matches fc_weight.mem's (out_features,in_features)
//   flatten order EXACTLY. This is the specific thing to double check here,
//   since a swapped in/out index order was suspected as a likely cause of
//   the previous project's stuck-at-one-class bug.
//
//   task05_export_weights.py exports the Dense layer via:
//     flatten_dense_weight(kernel) = np.transpose(kernel, (1,0)).flatten()
//   where Keras's kernel shape is (in_features, out_features). transpose(1,0)
//   reorders it to (out_features, in_features), and .flatten() is row-major,
//   i.e. the loop nesting is "for of: for inf:" - so the flat index is
//     weight_flat_index = of * IN_FEATURES + inf
//   This module's weight_index() function below computes EXACTLY that same
//   formula, and is the only place weight addresses are generated (no
//   address is ever computed ad hoc elsewhere), so there is a single
//   source of truth for this formula that matches the .mem layout.
//
//   Memory layout convention (matches global_avg_pool's ofmap convention,
//   so this module's ifmap port can read global_avg_pool's ofmap memory
//   directly - both are "one value per feature/channel", addr = index):
//     ifmap : addr = inf                      (IN_FEATURES entries)
//     weight: addr = of * IN_FEATURES + inf    (OUT_FEATURES*IN_FEATURES entries)
//     bias  : addr = of                        (OUT_FEATURES entries)
//     ofmap : addr = of                        (OUT_FEATURES entries)
//
//   Output width: ofmap_wdata is SCORE_W bits (default 16), NOT DATA_W (8)
//   like the other modules' activations. This is deliberate: this is the
//   final logit/score layer feeding argmax, and clipping class scores to
//   an 8-bit range could saturate multiple classes to the same value and
//   destroy the relative ordering argmax depends on. No saturation logic
//   exists in this module, so SCORE_W MUST be provisioned wide enough for
//   the worst case, as a function of THIS instance's parameters:
//     |acc|_max      <= IN_FEATURES * 128 * 128 + 2^(BIAS_W-1)
//     |acc >>> FRAC_BITS|_max = |acc|_max >> FRAC_BITS
//     required SCORE_W >= ceil(log2(|acc>>>FRAC_BITS|_max + 1)) + 1 (sign)
//   With the module defaults (IN_FEATURES=64, DATA_W=8, BIAS_W=16,
//   FRAC_BITS=7): |acc|_max ~= 1,048,608, shifted ~= 8192 -> needs ~15
//   bits -> the default SCORE_W=16 covers it.
//   BUT: FRAC_BITS is NOT always 7 (see above) - a SMALLER FRAC_BITS means
//   LESS right-shift, i.e. a LARGER shifted result. With FC_FRAC_BITS=3 (the
//   real model's fc layer, IN_FEATURES=64): shifted ~= 1,048,608 >> 3 ~=
//   131,076 -> needs ~19 bits, which the default SCORE_W=16 would silently
//   truncate/alias. This is exactly why cnn_core_fsm.v does NOT use fc.v's
//   default SCORE_W - it explicitly instantiates with SCORE_W=24 for
//   headroom. Any time FRAC_BITS, IN_FEATURES, DATA_W, or BIAS_W change at
//   an instantiation site, re-run this formula and re-check SCORE_W there,
//   not here - this module cannot know FRAC_BITS in advance since it is a
//   parameter, and there is no saturation to fall back on if the bound is
//   exceeded (see intro above for why saturating this layer is undesirable).
//
//   Same BRAM timing assumption and FETCH/COMPUTE/WRITE state-per-cycle
//   style as conv_layer: address held stable for one full cycle ->
//   registered read data valid the following cycle.
// ============================================================================
module fc #(
    parameter IN_FEATURES  = 64,
    parameter OUT_FEATURES = 9,
    parameter DATA_W    = 8,    // activation / weight width (frac_bits is external context, not fixed here)
    parameter BIAS_W    = 16,   // Q2.14 bias width
    parameter ACC_W     = 32,   // internal accumulator width
    parameter FRAC_BITS = 7,    // right-shift applied after bias-add
    parameter SCORE_W   = 16,   // output class-score width (wider than DATA_W - see header)
    parameter ADDR_W    = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  busy,
    output reg  done,

    // input activations (read-only) - IN_FEATURES entries
    output reg  [ADDR_W-1:0]        ifmap_addr,
    input  wire signed [DATA_W-1:0] ifmap_rdata,

    // weight memory (read-only) - OUT_FEATURES*IN_FEATURES entries, (of,inf) order
    output reg  [ADDR_W-1:0]        weight_addr,
    input  wire signed [DATA_W-1:0] weight_rdata,

    // bias memory (read-only) - OUT_FEATURES entries
    output reg  [ADDR_W-1:0]        bias_addr,
    input  wire signed [BIAS_W-1:0] bias_rdata,

    // output class scores (write-only) - OUT_FEATURES entries, no ReLU/saturate
    output reg                        ofmap_we,
    output reg  [ADDR_W-1:0]          ofmap_addr,
    output reg  signed [SCORE_W-1:0]  ofmap_wdata
);

    // ------------------------------------------------------------------
    // address helper functions (pure, no state) - single source of truth
    // for the weight flatten order described in the header comment.
    // ------------------------------------------------------------------
    function integer ifmap_index;
        input integer inf_f;
        begin
            ifmap_index = inf_f;
        end
    endfunction

    function integer weight_index;
        input integer of_f, inf_f;
        begin
            weight_index = of_f * IN_FEATURES + inf_f; // (out_features, in_features) order - matches fc_weight.mem
        end
    endfunction

    // ------------------------------------------------------------------
    // loop counters - plain 'integer' (32-bit signed), same rationale as
    // conv_layer/maxpool/global_avg_pool.
    // ------------------------------------------------------------------
    integer of, inf;
    integer next_inf;

    wire term_is_last  = (inf == IN_FEATURES-1);
    wire layer_is_last = (of == OUT_FEATURES-1);

    localparam S_IDLE      = 3'd0,
               S_BIAS_ADDR = 3'd1,   // bias_addr held stable this cycle
               S_BIAS_CAP  = 3'd2,   // bias_rdata now valid; also issues address for the first MAC term
               S_FETCH     = 3'd3,   // ifmap_addr/weight_addr held stable this cycle
               S_COMPUTE   = 3'd4,   // ifmap_rdata/weight_rdata now valid; do the MAC, issue next address
               S_WRITE     = 3'd5,   // shift (no ReLU/saturate), pulse ofmap_we, advance to next output feature
               S_DONE      = 3'd6;

    reg [2:0] state;
    reg signed [BIAS_W-1:0] bias_reg;
    reg signed [ACC_W-1:0]  acc;
    reg signed [ACC_W-1:0]  shifted_val; // S_WRITE scratch (blocking-assigned, same pattern as conv_layer)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            ofmap_we    <= 1'b0;
            of <= 0; inf <= 0;
            acc         <= {ACC_W{1'b0}};
            bias_reg    <= {BIAS_W{1'b0}};
            ifmap_addr  <= {ADDR_W{1'b0}};
            weight_addr <= {ADDR_W{1'b0}};
            bias_addr   <= {ADDR_W{1'b0}};
            ofmap_addr  <= {ADDR_W{1'b0}};
            ofmap_wdata <= {SCORE_W{1'b0}};
        end else begin
            ofmap_we <= 1'b0;
            done     <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy      <= 1'b1;
                    of <= 0; inf <= 0;
                    bias_addr <= 0;
                    state     <= S_BIAS_ADDR;
                end
            end

            // ----------------------------------------------------------
            S_BIAS_ADDR: begin
                state <= S_BIAS_CAP;
            end

            // ----------------------------------------------------------
            S_BIAS_CAP: begin
                bias_reg <= bias_rdata;
                acc      <= {{(ACC_W-BIAS_W){bias_rdata[BIAS_W-1]}}, bias_rdata}; // sign-extend Q2.14 bias into the accumulator

                ifmap_addr  <= ifmap_index(0);
                weight_addr <= weight_index(of, 0);
                state       <= S_FETCH;
            end

            // ----------------------------------------------------------
            S_FETCH: begin
                state <= S_COMPUTE;
            end

            // ----------------------------------------------------------
            S_COMPUTE: begin
                acc <= acc + ($signed(ifmap_rdata) * $signed(weight_rdata));

                if (term_is_last) begin
                    state <= S_WRITE;
                end else begin
                    next_inf = inf + 1;
                    inf <= next_inf;
                    ifmap_addr  <= ifmap_index(next_inf);
                    weight_addr <= weight_index(of, next_inf);
                    state <= S_FETCH;
                end
            end

            // ----------------------------------------------------------
            S_WRITE: begin
                // no ReLU, no saturate - see header comment for the margin
                // that makes truncation to SCORE_W bits exact
                shifted_val = acc >>> FRAC_BITS;

                ofmap_we    <= 1'b1;
                ofmap_addr  <= of;
                ofmap_wdata <= shifted_val[SCORE_W-1:0];

                inf <= 0;

                if (!layer_is_last) begin
                    of        <= of + 1;
                    bias_addr <= of + 1;
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
