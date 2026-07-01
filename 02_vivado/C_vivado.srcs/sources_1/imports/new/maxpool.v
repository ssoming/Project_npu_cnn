`timescale 1ns / 1ps
// ============================================================================
// maxpool
//   2x2 max pooling, stride 2, fixed window (not parameterized - only 2x2/
//   stride-2 is required by this project). Parameterized on IMG_W/IMG_H/
//   IN_CH so the same RTL is reused for pool1/pool2/pool3. Input spatial
//   size must be even (true for every layer here: 48->24, 24->12, 12->6).
//   Output size is exactly half: OUT_W=IMG_W/2, OUT_H=IMG_H/2.
//
//   Memory layout convention (matches conv_layer's ifmap/ofmap convention,
//   so conv_layer's ofmap memory can be read directly as this module's
//   ifmap without any reshaping):
//     ifmap : addr = (ic * IMG_H + iy) * IMG_W + ix   (channel-major plane, pre-pool size)
//     ofmap : addr = (ic * OUT_H + oy) * OUT_W + ox    (channel-major plane, post-pool size)
//
//   Same BRAM timing assumption as conv_layer: address held stable for one
//   full clock cycle -> registered read data valid the following cycle.
//   Same FETCH/COMPUTE(here: CAP)/WRITE state-per-cycle style as conv_layer,
//   for the same reason: avoid pipeline-overlap addressing bugs.
// ============================================================================
module maxpool #(
    parameter IMG_W  = 48,
    parameter IMG_H  = 48,
    parameter IN_CH  = 8,
    parameter DATA_W = 8,
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

    // output feature map (write-only) - post-pool size (OUT_W x OUT_H x IN_CH)
    output reg                      ofmap_we,
    output reg  [ADDR_W-1:0]        ofmap_addr,
    output reg  signed [DATA_W-1:0] ofmap_wdata
);

    localparam OUT_W = IMG_W / 2;
    localparam OUT_H = IMG_H / 2;

    // ------------------------------------------------------------------
    // address helper functions (pure, no state)
    // ------------------------------------------------------------------
    function integer ifmap_index;
        input integer ic_f, iy_f, ix_f;
        begin
            ifmap_index = (ic_f * IMG_H + iy_f) * IMG_W + ix_f;
        end
    endfunction

    function integer ofmap_index;
        input integer ic_f, oy_f, ox_f;
        begin
            ofmap_index = (ic_f * OUT_H + oy_f) * OUT_W + ox_f;
        end
    endfunction

    // ------------------------------------------------------------------
    // loop counters - plain 'integer' (32-bit signed), same rationale as
    // conv_layer: avoids unsigned-wrap surprises in address arithmetic.
    //   tap: which of the 4 window cells is being fetched
    //        0 = top-left (ty=0,tx=0), 1 = top-right (ty=0,tx=1)
    //        2 = bottom-left (ty=1,tx=0), 3 = bottom-right (ty=1,tx=1)
    // ------------------------------------------------------------------
    integer ic, oy, ox, tap;
    integer next_tap;

    wire tap_is_last   = (tap == 3);
    wire pixel_is_last = (ox == OUT_W-1) && (oy == OUT_H-1);
    wire layer_is_last = (ic == IN_CH-1);

    localparam S_IDLE  = 3'd0,
               S_FETCH = 3'd1,  // ifmap_addr held stable this cycle
               S_CAP   = 3'd2,  // ifmap_rdata now valid; update running max, issue next address
               S_WRITE = 3'd3,  // pulse ofmap_we, advance to next pixel/channel
               S_DONE  = 3'd4;

    reg [2:0] state;
    reg signed [DATA_W-1:0] max_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            ofmap_we    <= 1'b0;
            ic <= 0; oy <= 0; ox <= 0; tap <= 0;
            max_reg     <= {DATA_W{1'b0}};
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
                    ic <= 0; oy <= 0; ox <= 0; tap <= 0;
                    ifmap_addr <= ifmap_index(0, 0, 0);
                    state      <= S_FETCH;
                end
            end

            // ----------------------------------------------------------
            S_FETCH: begin
                // address set last cycle is now presented to the BRAM;
                // wait one cycle for synchronous read data to appear
                state <= S_CAP;
            end

            // ----------------------------------------------------------
            S_CAP: begin
                // ifmap_rdata now corresponds to the address issued during
                // S_FETCH (== current ic/oy/ox/tap)
                if (tap == 0)
                    max_reg <= ifmap_rdata;                              // first tap: unconditional load
                else if ($signed(ifmap_rdata) > $signed(max_reg))
                    max_reg <= ifmap_rdata;                              // signed compare (inputs may be negative)

                if (tap_is_last) begin
                    state <= S_WRITE;
                end else begin
                    next_tap   = tap + 1;
                    tap        <= next_tap;
                    // next_tap bit1 = ty offset (0 or 1), bit0 = tx offset (0 or 1)
                    ifmap_addr <= ifmap_index(ic, 2*oy + (next_tap >> 1), 2*ox + (next_tap & 1));
                    state      <= S_FETCH;
                end
            end

            // ----------------------------------------------------------
            S_WRITE: begin
                ofmap_we    <= 1'b1;
                ofmap_addr  <= ofmap_index(ic, oy, ox);
                ofmap_wdata <= max_reg;

                tap <= 0;

                if (!pixel_is_last) begin
                    if (ox == OUT_W-1) begin
                        ox <= 0;
                        oy <= oy + 1;
                        ifmap_addr <= ifmap_index(ic, 2*(oy+1), 0);
                    end else begin
                        ox <= ox + 1;
                        ifmap_addr <= ifmap_index(ic, 2*oy, 2*(ox+1));
                    end
                    state <= S_FETCH;

                end else if (!layer_is_last) begin
                    oy <= 0;
                    ox <= 0;
                    ic <= ic + 1;
                    ifmap_addr <= ifmap_index(ic+1, 0, 0);
                    state <= S_FETCH;

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
