`timescale 1ns / 1ps
// ============================================================================
// axi_lite_cnn_wrapper
//   AXI4-Lite slave wrapper around cnn_core_fsm (instantiated as-is, no
//   logic copy-pasted - see the instantiation near the bottom). Intended
//   system base address is 0x43C00000; that address is NOT hardcoded here
//   (per standard AXI-Lite peripheral practice) - it is assigned by the
//   AXI interconnect / Vivado Address Editor when this IP is added to a
//   block design. This module only decodes the LOCAL register offset
//   (0x00-0x14) within whatever base address it is given.
//
//   Register map (word-aligned, C_S_AXI_ADDR_WIDTH=5 -> 8 word slots):
//     0x00 CTRL         [0]=start (write-1-to-pulse; always reads 0)
//     0x04 STATUS       [0]=busy (live), [1]=done (LATCHED - see below)
//     0x08 SW_RESET     write-any-value-to-pulse (always reads 0)
//     0x0C CLASS_RESULT [3:0]=argmax result 0..8 (read-only, live)
//     0x10 IMG_DATA     [7:0]=one pixel value (write-only; write pulses
//                        img_we for exactly 1 cycle at IMG_ADDR's current
//                        value; always reads 0)
//     0x14 IMG_ADDR     [11:0]=pixel address 0..2303 (read/write)
//
//   ---- Careful design points (per the previous project's bug history) ----
//
//   1) STATUS.done is a LATCH, not a direct passthrough of cnn_core_fsm's
//      'done' output. cnn_core_fsm.done is only a single clock-cycle pulse
//      (by design, matching every sub-module's start/done handshake in
//      this project) - if software polls STATUS over AXI (which happens
//      far slower than one clock cycle), a raw passthrough would almost
//      certainly be missed entirely, making the peripheral look like it
//      never finishes. done_latch sets on cnn_core_fsm's done pulse and
//      stays set until the NEXT start is issued or SW_RESET is written,
//      so a polling loop reliably observes completion regardless of its
//      poll rate.
//
//   2) img_we and start are generated as single-cycle pulses tied to the
//      exact AXI write-accept cycle (slv_reg_wren), not to the write DATA
//      value being merely present or to a held register bit. This avoids
//      two failure modes: (a) writing img_we for more than one cycle would
//      write the same pixel into multiple/wrong addresses if IMG_ADDR
//      hadn't been advanced yet, and (b) a level-sensitive 'start' bit that
//      stayed high would re-trigger cnn_core_fsm repeatedly instead of
//      running exactly one inference per CTRL write. CTRL/IMG_DATA/SW_RESET
//      therefore always read back 0 - they are pure write-strobes, not
//      stored state, so there is nothing to accidentally leave "stuck on".
//
//   3) SW_RESET's reset scope is deliberately narrow: it resets ONLY
//      cnn_core_fsm's internal state (via cnn_rst_n) plus this wrapper's
//      done_latch (so a stale done doesn't survive a reset) and img_addr
//      (clean slate for the next image load). It does NOT reset the AXI
//      protocol state machine (axi_awready/axi_wready/axi_bvalid/
//      axi_arready/axi_rvalid) - those are governed solely by
//      s_axi_aresetn. If SW_RESET reset the AXI state machine too, writing
//      to SW_RESET would corrupt the very same AXI write transaction that
//      triggered it (the write-response phase would be reset mid-flight).
//      The reset pulse to cnn_core_fsm is also stretched to a few cycles
//      (not exactly 1) to be robust against reset-pulse-width edge cases,
//      since cnn_core_fsm's reset is asynchronous-assert.
//
//   ---- AXI4-Lite slave protocol logic ----
//   This follows the standard single-outstanding-transaction AXI4-Lite
//   slave template (the same shape Vivado's own AXI4-Lite peripheral
//   wizard generates): AWREADY/WREADY pulse together for one cycle once
//   both AWVALID and WVALID are seen (arbitrated by aw_en so a new address
//   isn't accepted until the previous write's BVALID/BREADY has completed),
//   the actual register write happens on that same "slv_reg_wren" cycle,
//   and BVALID/RVALID follow the standard request->response handshake.
//
//   Pure Verilog-2001 (no SystemVerilog constructs), consistent with every
//   other file in this project.
// ============================================================================
module axi_lite_cnn_wrapper #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 5,   // 8 word-aligned slots: 0x00..0x1C (regs used: 0x00-0x14)

    // --- pass-through parameters for the cnn_core_fsm instance below ---
    // defaults match the real trained model; a testbench may override all
    // of these to a small size (with matching small-random weight .mem
    // files) for a fast AXI-protocol-only test, without touching this
    // module's logic.
    parameter IMG0_W      = 48,
    parameter IMG0_H      = 48,
    parameter C0          = 1,
    parameter C1          = 8,
    parameter C2          = 16,
    parameter C3          = 64,
    parameter NUM_CLASSES = 9,
    parameter DATA_W    = 8,
    parameter BIAS_W    = 16,
    parameter ACC_W     = 32,
    parameter SCORE_W   = 24,
    parameter ADDR_W    = 16,
    parameter CONV1_FRAC_BITS = 6,
    parameter CONV2_FRAC_BITS = 7,
    parameter CONV3_FRAC_BITS = 7,
    parameter FC_FRAC_BITS    = 3,
    parameter CONV1_WEIGHT_FILE = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/conv1_weight.mem",
    parameter CONV1_BIAS_FILE   = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/conv1_bias.mem",
    parameter CONV2_WEIGHT_FILE = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/conv2_weight.mem",
    parameter CONV2_BIAS_FILE   = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/conv2_bias.mem",
    parameter CONV3_WEIGHT_FILE = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/conv3_weight.mem",
    parameter CONV3_BIAS_FILE   = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/conv3_bias.mem",
    parameter FC_WEIGHT_FILE    = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/fc_weight.mem",
    parameter FC_BIAS_FILE      = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/fc_bias.mem"
)(
    // ---- standard AXI4-Lite slave port names (for Vivado IP packaging) ----
    input  wire                              s_axi_aclk,
    input  wire                              s_axi_aresetn,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire [2:0]                        s_axi_awprot,
    input  wire                              s_axi_awvalid,
    output wire                              s_axi_awready,

    input  wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                              s_axi_wvalid,
    output wire                              s_axi_wready,

    output wire [1:0]                        s_axi_bresp,
    output wire                              s_axi_bvalid,
    input  wire                              s_axi_bready,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire [2:0]                        s_axi_arprot,
    input  wire                              s_axi_arvalid,
    output wire                              s_axi_arready,

    output wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output wire [1:0]                        s_axi_rresp,
    output wire                              s_axi_rvalid,
    input  wire                              s_axi_rready
);

    // ------------------------------------------------------------------
    // register offsets (word index = addr[4:2])
    // ------------------------------------------------------------------
    localparam REG_CTRL         = 3'h0; // 0x00
    localparam REG_STATUS       = 3'h1; // 0x04
    localparam REG_SW_RESET     = 3'h2; // 0x08
    localparam REG_CLASS_RESULT = 3'h3; // 0x0C
    localparam REG_IMG_DATA     = 3'h4; // 0x10
    localparam REG_IMG_ADDR     = 3'h5; // 0x14

    // ------------------------------------------------------------------
    // AXI4-Lite write channel (standard template)
    // ------------------------------------------------------------------
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg                          axi_awready;
    reg                          axi_wready;
    reg [1:0]                    axi_bresp;
    reg                          axi_bvalid;
    reg                          aw_en;

    assign s_axi_awready = axi_awready;
    assign s_axi_wready  = axi_wready;
    assign s_axi_bresp   = axi_bresp;
    assign s_axi_bvalid  = axi_bvalid;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_awready <= 1'b0;
            aw_en       <= 1'b1;
        end else if (~axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
            axi_awready <= 1'b1;
            aw_en       <= 1'b0;
        end else if (s_axi_bready && axi_bvalid) begin
            aw_en       <= 1'b1;
            axi_awready <= 1'b0;
        end else begin
            axi_awready <= 1'b0;
        end
    end

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_awaddr <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else if (~axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
            axi_awaddr <= s_axi_awaddr;
        end
    end

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_wready <= 1'b0;
        end else if (~axi_wready && s_axi_wvalid && s_axi_awvalid && aw_en) begin
            axi_wready <= 1'b1;
        end else begin
            axi_wready <= 1'b0;
        end
    end

    // exactly one cycle wide: true only on the cycle both AW and W are accepted
    wire slv_reg_wren = axi_wready && s_axi_wvalid && axi_awready && s_axi_awvalid;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_bvalid <= 1'b0;
            axi_bresp  <= 2'b00;
        end else if (slv_reg_wren) begin
            axi_bvalid <= 1'b1;
            axi_bresp  <= 2'b00; // OKAY
        end else if (s_axi_bready && axi_bvalid) begin
            axi_bvalid <= 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // AXI4-Lite read channel (standard template)
    // ------------------------------------------------------------------
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg                          axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    reg [1:0]                    axi_rresp;
    reg                          axi_rvalid;

    assign s_axi_arready = axi_arready;
    assign s_axi_rdata   = axi_rdata;
    assign s_axi_rresp   = axi_rresp;
    assign s_axi_rvalid  = axi_rvalid;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_arready <= 1'b0;
            axi_araddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else if (~axi_arready && s_axi_arvalid) begin
            axi_arready <= 1'b1;
            axi_araddr  <= s_axi_araddr;
        end else begin
            axi_arready <= 1'b0;
        end
    end

    wire slv_reg_rden = axi_arready && s_axi_arvalid && ~axi_rvalid;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_rvalid <= 1'b0;
            axi_rresp  <= 2'b00;
        end else if (slv_reg_rden) begin
            axi_rvalid <= 1'b1;
            axi_rresp  <= 2'b00; // OKAY
        end else if (axi_rvalid && s_axi_rready) begin
            axi_rvalid <= 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // register write decode - generates the start/img_we/sw_reset pulses
    // and stores img_addr (see header comments 1-3 for the reasoning)
    // ------------------------------------------------------------------
    reg                      start_pulse;
    reg                      sw_reset_pulse;
    reg                      img_we_pulse;
    reg  [11:0]              img_addr_reg;
    reg  signed [DATA_W-1:0] img_wdata_reg;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            start_pulse    <= 1'b0;
            sw_reset_pulse <= 1'b0;
            img_we_pulse   <= 1'b0;
            img_addr_reg   <= 12'd0;
            img_wdata_reg  <= {DATA_W{1'b0}};
        end else begin
            // defaults: every pulse below is exactly 1 cycle wide unless
            // re-asserted by a matching write this cycle
            start_pulse    <= 1'b0;
            sw_reset_pulse <= 1'b0;
            img_we_pulse   <= 1'b0;

            if (slv_reg_wren) begin
                case (axi_awaddr[4:2])
                    REG_CTRL: begin
                        if (s_axi_wstrb[0] && s_axi_wdata[0])
                            start_pulse <= 1'b1;
                    end
                    REG_SW_RESET: begin
                        sw_reset_pulse <= 1'b1;
                    end
                    REG_IMG_DATA: begin
                        img_wdata_reg <= s_axi_wdata[DATA_W-1:0];
                        img_we_pulse  <= 1'b1;
                    end
                    REG_IMG_ADDR: begin
                        if (s_axi_wstrb[0]) img_addr_reg[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) img_addr_reg[11:8] <= s_axi_wdata[11:8];
                    end
                    default: ; // STATUS/CLASS_RESULT are read-only, writes ignored
                endcase
            end

            // SW_RESET also clears img_addr for a clean slate on the next image load
            if (sw_reset_pulse) img_addr_reg <= 12'd0;
        end
    end

    // ------------------------------------------------------------------
    // cnn_core_fsm output wires - declared here (before use below) since
    // plain Verilog-2001 requires declaration to textually precede use;
    // the driving instance is at the bottom of this file.
    // ------------------------------------------------------------------
    wire        cnn_busy;
    wire        cnn_done;
    wire [3:0]  cnn_class_idx;

    // ------------------------------------------------------------------
    // STATUS.done latch (see header comment 1) and SW_RESET scope/stretch
    // (see header comment 3)
    // ------------------------------------------------------------------
    reg done_latch;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            done_latch <= 1'b0;
        end else if (start_pulse || sw_reset_pulse) begin
            done_latch <= 1'b0;
        end else if (cnn_done) begin
            done_latch <= 1'b1;
        end
    end

    reg [2:0] sw_reset_stretch;
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn)
            sw_reset_stretch <= 3'b000;
        else if (sw_reset_pulse)
            sw_reset_stretch <= 3'b111;
        else
            sw_reset_stretch <= {1'b0, sw_reset_stretch[2:1]};
    end
    wire cnn_rst_n = s_axi_aresetn & ~(|sw_reset_stretch);

    // ------------------------------------------------------------------
    // register read mux
    // ------------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}};
        end else if (slv_reg_rden) begin
            case (axi_araddr[4:2])
                REG_CTRL:         axi_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}}; // write-strobe only
                REG_STATUS:       axi_rdata <= {{(C_S_AXI_DATA_WIDTH-2){1'b0}}, done_latch, cnn_busy};
                REG_SW_RESET:     axi_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}}; // write-strobe only
                REG_CLASS_RESULT: axi_rdata <= {{(C_S_AXI_DATA_WIDTH-4){1'b0}}, cnn_class_idx};
                REG_IMG_DATA:     axi_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}}; // write-only
                REG_IMG_ADDR:     axi_rdata <= {{(C_S_AXI_DATA_WIDTH-12){1'b0}}, img_addr_reg};
                default:          axi_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}};
            endcase
        end
    end

    // ------------------------------------------------------------------
    // cnn_core_fsm instance - reused as-is, no logic copied
    // ------------------------------------------------------------------
    cnn_core_fsm #(
        .IMG0_W(IMG0_W), .IMG0_H(IMG0_H), .C0(C0), .C1(C1), .C2(C2), .C3(C3), .NUM_CLASSES(NUM_CLASSES),
        .DATA_W(DATA_W), .BIAS_W(BIAS_W), .ACC_W(ACC_W), .SCORE_W(SCORE_W), .ADDR_W(ADDR_W),
        .CONV1_FRAC_BITS(CONV1_FRAC_BITS), .CONV2_FRAC_BITS(CONV2_FRAC_BITS),
        .CONV3_FRAC_BITS(CONV3_FRAC_BITS), .FC_FRAC_BITS(FC_FRAC_BITS),
        .CONV1_WEIGHT_FILE(CONV1_WEIGHT_FILE), .CONV1_BIAS_FILE(CONV1_BIAS_FILE),
        .CONV2_WEIGHT_FILE(CONV2_WEIGHT_FILE), .CONV2_BIAS_FILE(CONV2_BIAS_FILE),
        .CONV3_WEIGHT_FILE(CONV3_WEIGHT_FILE), .CONV3_BIAS_FILE(CONV3_BIAS_FILE),
        .FC_WEIGHT_FILE(FC_WEIGHT_FILE), .FC_BIAS_FILE(FC_BIAS_FILE)
    ) cnn_inst (
        .clk(s_axi_aclk),
        .rst_n(cnn_rst_n),
        .start(start_pulse),
        .busy(cnn_busy),
        .done(cnn_done),
        .class_idx(cnn_class_idx),
        .img_we(img_we_pulse),
        .img_addr({{(ADDR_W-12){1'b0}}, img_addr_reg}), // zero-extend 12-bit register addr into cnn_core_fsm's ADDR_W-bit port
        .img_wdata(img_wdata_reg)
    );

endmodule
