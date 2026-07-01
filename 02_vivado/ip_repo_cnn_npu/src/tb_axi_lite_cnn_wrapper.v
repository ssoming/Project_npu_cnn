`timescale 1ns / 1ps
// ============================================================================
// tb_axi_lite_cnn_wrapper - self-checking AXI4-Lite protocol test
//   Pure Verilog-2001 (no SystemVerilog constructs), same convention as
//   every other testbench in this project.
//
//   This is a PROTOCOL test (does register read/write, start/done
//   handshake, image loading via IMG_ADDR/IMG_DATA, and SW_RESET behave
//   correctly over AXI4-Lite) - NOT a re-verification of conv/pool/gap/fc
//   math (already proven in their own unit tests) or of cnn_core_fsm's
//   8-stage sequencing (already proven 9/9 on real images). To keep
//   simulation time reasonable, the DUT is instantiated with the same
//   small size used for cnn_core_fsm's own wiring smoke test (16x16x1 in,
//   4/6/8 conv channels, 3 classes) and small-random weight .mem files at
//   python/python/mem/tb_small_test/, instead of the real 48x48/.../9
//   default. Golden class_idx=1 was recomputed in Python using the exact
//   same per-layer weight_frac_bits cnn_core_fsm now defaults to (conv1=6,
//   conv2=7, conv3=7, fc=3).
//
//   Register map under test (see axi_lite_cnn_wrapper.v header):
//     0x00 CTRL, 0x04 STATUS, 0x08 SW_RESET, 0x0C CLASS_RESULT,
//     0x10 IMG_DATA, 0x14 IMG_ADDR
//
//   Cases:
//     1: image load (256x IMG_ADDR/IMG_DATA writes) -> CTRL.start=1 ->
//        poll STATUS.done -> read CLASS_RESULT, compare to golden
//     2: IMG_ADDR read-back (write then read the same value back)
//     3: SW_RESET clears STATUS (busy=0, done=0) and cnn_core_fsm can run
//        a fresh, correct inference afterward (not stuck/corrupted)
// ============================================================================
module tb_axi_lite_cnn_wrapper;

    localparam C_S_AXI_DATA_WIDTH = 32;
    localparam C_S_AXI_ADDR_WIDTH = 5;

    localparam IMG0_W = 16;
    localparam IMG0_H = 16;
    localparam C0 = 1;
    localparam C1 = 4;
    localparam C2 = 6;
    localparam C3 = 8;
    localparam NUM_CLASSES = 3;
    localparam DATA_W = 8;

    localparam IMG_N = IMG0_W*IMG0_H*C0; // 256
    localparam GOLDEN_CLASS = 1;

    localparam MEM_DIR = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/tb_small_test/";

    // register addresses
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_CTRL         = 5'h00;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_STATUS       = 5'h04;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_SW_RESET     = 5'h08;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_CLASS_RESULT = 5'h0C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_IMG_DATA     = 5'h10;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_IMG_ADDR     = 5'h14;

    reg clk, aresetn;

    reg  [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr;
    reg  [2:0]                        s_axi_awprot;
    reg                               s_axi_awvalid;
    wire                              s_axi_awready;
    reg  [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata;
    reg  [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb;
    reg                               s_axi_wvalid;
    wire                              s_axi_wready;
    wire [1:0]                        s_axi_bresp;
    wire                              s_axi_bvalid;
    reg                               s_axi_bready;
    reg  [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr;
    reg  [2:0]                        s_axi_arprot;
    reg                               s_axi_arvalid;
    wire                              s_axi_arready;
    wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata;
    wire [1:0]                        s_axi_rresp;
    wire                              s_axi_rvalid;
    reg                               s_axi_rready;

    axi_lite_cnn_wrapper #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH), .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .IMG0_W(IMG0_W), .IMG0_H(IMG0_H), .C0(C0), .C1(C1), .C2(C2), .C3(C3), .NUM_CLASSES(NUM_CLASSES),
        .CONV1_WEIGHT_FILE({MEM_DIR, "conv1_weight.mem"}), .CONV1_BIAS_FILE({MEM_DIR, "conv1_bias.mem"}),
        .CONV2_WEIGHT_FILE({MEM_DIR, "conv2_weight.mem"}), .CONV2_BIAS_FILE({MEM_DIR, "conv2_bias.mem"}),
        .CONV3_WEIGHT_FILE({MEM_DIR, "conv3_weight.mem"}), .CONV3_BIAS_FILE({MEM_DIR, "conv3_bias.mem"}),
        .FC_WEIGHT_FILE({MEM_DIR, "fc_weight.mem"}),       .FC_BIAS_FILE({MEM_DIR, "fc_bias.mem"})
    ) dut (
        .s_axi_aclk(clk), .s_axi_aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awprot(s_axi_awprot), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arprot(s_axi_arprot), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready)
    );

    always #5 clk = ~clk; // 100MHz

    // ------------------------------------------------------------------
    // minimal AXI4-Lite master BFM
    // ------------------------------------------------------------------
    task axi_write(input [C_S_AXI_ADDR_WIDTH-1:0] addr, input [C_S_AXI_DATA_WIDTH-1:0] data);
        begin
            s_axi_awaddr  = addr;
            s_axi_awvalid = 1'b1;
            s_axi_wdata   = data;
            s_axi_wstrb   = 4'hF;
            s_axi_wvalid  = 1'b1;
            s_axi_bready  = 1'b1;
            @(posedge clk);
            while (!(s_axi_awready && s_axi_wready)) @(posedge clk);
            s_axi_awvalid = 1'b0;
            s_axi_wvalid  = 1'b0;
            while (!s_axi_bvalid) @(posedge clk);
            @(posedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task axi_read(input [C_S_AXI_ADDR_WIDTH-1:0] addr, output [C_S_AXI_DATA_WIDTH-1:0] data);
        begin
            s_axi_araddr  = addr;
            s_axi_arvalid = 1'b1;
            s_axi_rready  = 1'b1;
            @(posedge clk);
            while (!s_axi_arready) @(posedge clk);
            s_axi_arvalid = 1'b0;
            while (!s_axi_rvalid) @(posedge clk);
            data = s_axi_rdata;
            @(posedge clk);
            s_axi_rready = 1'b0;
        end
    endtask

    // ------------------------------------------------------------------
    // test image (matches cnn_core_fsm's original small wiring smoke test)
    // ------------------------------------------------------------------
    reg signed [DATA_W-1:0] image_mem [0:IMG_N-1];
    initial $readmemh({MEM_DIR, "image.mem"}, image_mem);

    integer i;
    reg [C_S_AXI_DATA_WIDTH-1:0] rdata;
    integer pass_count;
    integer fail_count;

    task run_inference;
        begin
            // load image via the register interface: write IMG_ADDR then
            // IMG_DATA for each pixel, exactly the flow spec'd for software
            for (i = 0; i < IMG_N; i = i + 1) begin
                axi_write(ADDR_IMG_ADDR, {{(C_S_AXI_DATA_WIDTH-12){1'b0}}, i[11:0]});
                axi_write(ADDR_IMG_DATA, {{(C_S_AXI_DATA_WIDTH-DATA_W){1'b0}}, image_mem[i]});
            end

            // start
            axi_write(ADDR_CTRL, 32'h1);

            // poll STATUS until done (bit1) is set
            rdata = 0;
            while (rdata[1] !== 1'b1) begin
                axi_read(ADDR_STATUS, rdata);
            end
        end
    endtask

    initial begin
        clk = 0; aresetn = 0;
        s_axi_awaddr=0; s_axi_awprot=0; s_axi_awvalid=0;
        s_axi_wdata=0; s_axi_wstrb=0; s_axi_wvalid=0;
        s_axi_bready=0;
        s_axi_araddr=0; s_axi_arprot=0; s_axi_arvalid=0;
        s_axi_rready=0;
        pass_count = 0;
        fail_count = 0;

        repeat (5) @(posedge clk);
        aresetn = 1;
        repeat (3) @(posedge clk);

        // ---------------------------------------------------------------
        // TEST 1: full image-load -> start -> poll done -> CLASS_RESULT
        // ---------------------------------------------------------------
        run_inference;
        axi_read(ADDR_CLASS_RESULT, rdata);
        if (rdata[3:0] === GOLDEN_CLASS[3:0]) begin
            pass_count = pass_count + 1;
            $display("[TEST 1] PASS  | image load + start + poll done + CLASS_RESULT | expected=%0d, got=%0d",
                      GOLDEN_CLASS, rdata[3:0]);
        end else begin
            fail_count = fail_count + 1;
            $display("[TEST 1] FAIL  | image load + start + poll done + CLASS_RESULT | expected=%0d, got=%0d",
                      GOLDEN_CLASS, rdata[3:0]);
        end

        // STATUS should show busy=0 (inference finished) at this point
        axi_read(ADDR_STATUS, rdata);
        if (rdata[0] === 1'b0) begin
            pass_count = pass_count + 1;
            $display("[TEST 1b] PASS | STATUS.busy==0 after done observed");
        end else begin
            fail_count = fail_count + 1;
            $display("[TEST 1b] FAIL | STATUS.busy==0 after done observed | got busy=%0d", rdata[0]);
        end

        // ---------------------------------------------------------------
        // TEST 2: IMG_ADDR read-back
        // ---------------------------------------------------------------
        axi_write(ADDR_IMG_ADDR, 32'h0000_02A5); // arbitrary 12-bit value within range (677)
        axi_read(ADDR_IMG_ADDR, rdata);
        if (rdata[11:0] === 12'h2A5) begin
            pass_count = pass_count + 1;
            $display("[TEST 2] PASS  | IMG_ADDR readback | wrote=0x2A5, got=0x%03x", rdata[11:0]);
        end else begin
            fail_count = fail_count + 1;
            $display("[TEST 2] FAIL  | IMG_ADDR readback | wrote=0x2A5, got=0x%03x", rdata[11:0]);
        end

        // ---------------------------------------------------------------
        // TEST 3: SW_RESET clears STATUS (busy=0, done=0) ...
        // ---------------------------------------------------------------
        axi_write(ADDR_SW_RESET, 32'h1);
        repeat (5) @(posedge clk); // let the reset stretch/settle
        axi_read(ADDR_STATUS, rdata);
        if (rdata[1:0] === 2'b00) begin
            pass_count = pass_count + 1;
            $display("[TEST 3] PASS  | SW_RESET clears STATUS (busy=0,done=0) | got STATUS=0x%08x", rdata);
        end else begin
            fail_count = fail_count + 1;
            $display("[TEST 3] FAIL  | SW_RESET clears STATUS (busy=0,done=0) | got STATUS=0x%08x", rdata);
        end

        // ... and cnn_core_fsm still runs a fresh, CORRECT inference afterward
        // (proves SW_RESET didn't corrupt the AXI protocol state machine or
        // leave cnn_core_fsm in a broken state - see axi_lite_cnn_wrapper.v
        // header comment 3 on SW_RESET scope)
        run_inference;
        axi_read(ADDR_CLASS_RESULT, rdata);
        if (rdata[3:0] === GOLDEN_CLASS[3:0]) begin
            pass_count = pass_count + 1;
            $display("[TEST 3b] PASS | inference after SW_RESET still correct | expected=%0d, got=%0d",
                      GOLDEN_CLASS, rdata[3:0]);
        end else begin
            fail_count = fail_count + 1;
            $display("[TEST 3b] FAIL | inference after SW_RESET still correct | expected=%0d, got=%0d",
                      GOLDEN_CLASS, rdata[3:0]);
        end

        $display("");
        if (fail_count == 0)
            $display("=== SUMMARY: %0d / %0d PASS (100%%) ===", pass_count, pass_count);
        else
            $display("=== SUMMARY: %0d / %0d PASS (%0d%%) ===", pass_count, pass_count+fail_count,
                      (pass_count*100)/(pass_count+fail_count));

        $finish;
    end

    initial begin
        #20_000_000; // small (16x16) config - fast, generous margin
        $display("=== TIMEOUT: simulation did not finish in time ===");
        $finish;
    end

endmodule
