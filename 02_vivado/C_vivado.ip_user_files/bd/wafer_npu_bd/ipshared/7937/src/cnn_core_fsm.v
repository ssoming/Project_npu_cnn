`timescale 1ns / 1ps
// ============================================================================
// cnn_core_fsm
//   Top-level sequencer for: conv1->pool1->conv2->pool2->conv3->pool3->
//   gap->fc->argmax. Reuses the four already-verified modules (conv_layer,
//   maxpool, global_avg_pool, fc) as separate named instances - no MAC/pool
//   logic is re-implemented here. This module only does two things:
//     1. Own the activation buffers and weight/bias ROMs between stages,
//        wired so each buffer's writer and reader agree on the same
//        channel-major planar address convention (see each sub-module's
//        header comment).
//     2. Sequence the 8 sub-module start/done handshakes plus a final
//        argmax scan, one stage at a time (S_X_START pulses start for one
//        cycle, S_X_WAIT holds until that stage's done pulses).
//
//   Buffer map (see the design-overview message for full derivation):
//     img_mem        : IMG0_W x IMG0_H x C0        (external write port + conv1 read)
//     conv1_out_mem  : IMG0_W x IMG0_H x C1         (conv1 write, pool1 read)
//     pool1_out_mem  : POOL1_W x POOL1_H x C1       (pool1 write, conv2 read)
//     conv2_out_mem  : POOL1_W x POOL1_H x C2       (conv2 write, pool2 read)
//     pool2_out_mem  : POOL2_W x POOL2_H x C2       (pool2 write, conv3 read)
//     conv3_out_mem  : POOL2_W x POOL2_H x C3       (conv3 write, pool3 read)
//     pool3_out_mem  : POOL3_W x POOL3_H x C3       (pool3 write, gap read)
//     gap_out_mem    : C3                            (gap write, fc read)
//     fc_out_mem     : NUM_CLASSES (SCORE_W-wide)    (fc write, argmax read)
//   Each buffer is its own dedicated BRAM array - no shared/offset address
//   space, so there is no possibility of one stage's addressing overrunning
//   into another stage's data (a likely-suspect bug class in the previous
//   version's integration).
//
//   Weight/bias ROMs are loaded via $readmemh from the file path parameters
//   below (defaults point at the real trained-weight .mem files). Because
//   $readmemh runs at elaboration/simulation start, these also become the
//   BRAM's synthesized initial contents when this design is implemented -
//   no runtime weight-loading path is required for correctness (AXI-Lite,
//   step 6, only needs to handle start/done/class_idx and image loading).
//
//   Pure Verilog-2001 (no SystemVerilog constructs), consistent with every
//   other file in this project.
// ============================================================================
module cnn_core_fsm #(
    parameter IMG0_W      = 48,
    parameter IMG0_H      = 48,
    parameter C0          = 1,   // input channels
    parameter C1          = 8,   // conv1 output channels
    parameter C2          = 16,  // conv2 output channels
    parameter C3          = 64,  // conv3 output channels
    parameter NUM_CLASSES = 9,

    parameter DATA_W    = 8,
    parameter BIAS_W    = 16,
    parameter ACC_W     = 32,
    parameter SCORE_W   = 24,  // must cover |acc>>>FC_FRAC_BITS|; FC_FRAC_BITS=3 (small shift, real model) needs ~19 bits - see fc.v header
    parameter ADDR_W    = 16,

    // per-layer weight right-shift amount (== that layer's weight_frac_bits
    // in quant_meta.json; bias_frac_bits = ACT_FRAC_BITS + this value, where
    // ACT_FRAC_BITS is the system-wide activation frac_bits fixed in
    // task05_export_weights.py - currently 2, not the DATA_W=8 Q1.7 (frac=7)
    // originally assumed; see the writeup in tb_cnn_core_fsm.v for why: real
    // intermediate activations reach ~16 in magnitude, so Q1.7 (range
    // 0..0.992) was silently saturating 12-18% of every layer's activations
    // and caused a class-collapse bug - fixed purely in the export script by
    // lowering ACT_FRAC_BITS, no RTL or port-width change needed).
    // Each layer's WEIGHTS separately get their own quantization scale
    // chosen by task05_export_weights.py to avoid clipping (e.g. fc's
    // weights have |w|max=9.14, so fc uses frac_bits=3 instead of a shared
    // 7 - using one shared frac=7 for every layer's weights, as an earlier
    // version of this file did, clipped ~25% of the fc weights; that was a
    // separate, smaller issue from the activation one above and was fixed
    // first, though it alone did not resolve the class-collapse). These
    // defaults must be re-checked against quant_meta.json's weight_frac_bits
    // (and the activation_format.frac_bits) any time the model is retrained
    // and re-exported.
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
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    output reg  busy,
    output reg  done,
    output reg  [3:0] class_idx,

    // input image load port (drive while busy==0, before pulsing start)
    input  wire                     img_we,
    input  wire [ADDR_W-1:0]        img_addr,
    input  wire signed [DATA_W-1:0] img_wdata
);

    // ------------------------------------------------------------------
    // derived spatial sizes ("same" padding conv keeps size; 2x2/stride2
    // pool exactly halves it)
    // ------------------------------------------------------------------
    localparam POOL1_W = IMG0_W / 2;
    localparam POOL1_H = IMG0_H / 2;
    localparam POOL2_W = POOL1_W / 2;
    localparam POOL2_H = POOL1_H / 2;
    localparam POOL3_W = POOL2_W / 2;
    localparam POOL3_H = POOL2_H / 2;

    // ------------------------------------------------------------------
    // buffer / ROM depths
    // ------------------------------------------------------------------
    localparam IMG_MEM_N      = IMG0_W  * IMG0_H  * C0;
    localparam CONV1_OUT_N    = IMG0_W  * IMG0_H  * C1;
    localparam POOL1_OUT_N    = POOL1_W * POOL1_H * C1;
    localparam CONV2_OUT_N    = POOL1_W * POOL1_H * C2;
    localparam POOL2_OUT_N    = POOL2_W * POOL2_H * C2;
    localparam CONV3_OUT_N    = POOL2_W * POOL2_H * C3;
    localparam POOL3_OUT_N    = POOL3_W * POOL3_H * C3;
    localparam GAP_OUT_N      = C3;
    localparam FC_OUT_N       = NUM_CLASSES;

    localparam CONV1_W_N = C1 * C0 * 9;
    localparam CONV1_B_N = C1;
    localparam CONV2_W_N = C2 * C1 * 9;
    localparam CONV2_B_N = C2;
    localparam CONV3_W_N = C3 * C2 * 9;
    localparam CONV3_B_N = C3;
    localparam FC_W_N    = NUM_CLASSES * C3;
    localparam FC_B_N    = NUM_CLASSES;

    // ------------------------------------------------------------------
    // activation buffers (each: one writer stage, one reader stage)
    // ------------------------------------------------------------------
    reg signed [DATA_W-1:0]  img_mem       [0:IMG_MEM_N-1];
    reg signed [DATA_W-1:0]  conv1_out_mem [0:CONV1_OUT_N-1];
    reg signed [DATA_W-1:0]  pool1_out_mem [0:POOL1_OUT_N-1];
    reg signed [DATA_W-1:0]  conv2_out_mem [0:CONV2_OUT_N-1];
    reg signed [DATA_W-1:0]  pool2_out_mem [0:POOL2_OUT_N-1];
    reg signed [DATA_W-1:0]  conv3_out_mem [0:CONV3_OUT_N-1];
    reg signed [DATA_W-1:0]  pool3_out_mem [0:POOL3_OUT_N-1];
    reg signed [DATA_W-1:0]  gap_out_mem   [0:GAP_OUT_N-1];
    reg signed [SCORE_W-1:0] fc_out_mem    [0:FC_OUT_N-1];

    // ------------------------------------------------------------------
    // weight / bias ROMs
    // ------------------------------------------------------------------
    reg signed [DATA_W-1:0] conv1_weight_mem [0:CONV1_W_N-1];
    reg signed [BIAS_W-1:0] conv1_bias_mem   [0:CONV1_B_N-1];
    reg signed [DATA_W-1:0] conv2_weight_mem [0:CONV2_W_N-1];
    reg signed [BIAS_W-1:0] conv2_bias_mem   [0:CONV2_B_N-1];
    reg signed [DATA_W-1:0] conv3_weight_mem [0:CONV3_W_N-1];
    reg signed [BIAS_W-1:0] conv3_bias_mem   [0:CONV3_B_N-1];
    reg signed [DATA_W-1:0] fc_weight_mem    [0:FC_W_N-1];
    reg signed [BIAS_W-1:0] fc_bias_mem      [0:FC_B_N-1];

    initial $readmemh(CONV1_WEIGHT_FILE, conv1_weight_mem);
    initial $readmemh(CONV1_BIAS_FILE,   conv1_bias_mem);
    initial $readmemh(CONV2_WEIGHT_FILE, conv2_weight_mem);
    initial $readmemh(CONV2_BIAS_FILE,   conv2_bias_mem);
    initial $readmemh(CONV3_WEIGHT_FILE, conv3_weight_mem);
    initial $readmemh(CONV3_BIAS_FILE,   conv3_bias_mem);
    initial $readmemh(FC_WEIGHT_FILE,    fc_weight_mem);
    initial $readmemh(FC_BIAS_FILE,      fc_bias_mem);

    // ------------------------------------------------------------------
    // img_mem: external write port + conv1 read port (simple dual-port)
    // ------------------------------------------------------------------
    always @(posedge clk) if (img_we) img_mem[img_addr] <= img_wdata;

    // ------------------------------------------------------------------
    // conv1
    // ------------------------------------------------------------------
    reg  conv1_start;
    wire conv1_busy, conv1_done;
    wire [ADDR_W-1:0] conv1_ifmap_addr;
    reg  signed [DATA_W-1:0] conv1_ifmap_rdata;
    wire [ADDR_W-1:0] conv1_weight_addr;
    reg  signed [DATA_W-1:0] conv1_weight_rdata;
    wire [ADDR_W-1:0] conv1_bias_addr;
    reg  signed [BIAS_W-1:0] conv1_bias_rdata;
    wire conv1_ofmap_we;
    wire [ADDR_W-1:0] conv1_ofmap_addr;
    wire signed [DATA_W-1:0] conv1_ofmap_wdata;

    always @(posedge clk) conv1_ifmap_rdata  <= img_mem[conv1_ifmap_addr];
    always @(posedge clk) conv1_weight_rdata <= conv1_weight_mem[conv1_weight_addr];
    always @(posedge clk) conv1_bias_rdata   <= conv1_bias_mem[conv1_bias_addr];
    always @(posedge clk) if (conv1_ofmap_we) conv1_out_mem[conv1_ofmap_addr] <= conv1_ofmap_wdata;

    conv_layer #(
        .IMG_W(IMG0_W), .IMG_H(IMG0_H), .IN_CH(C0), .OUT_CH(C1), .KSIZE(3),
        .DATA_W(DATA_W), .BIAS_W(BIAS_W), .ACC_W(ACC_W), .FRAC_BITS(CONV1_FRAC_BITS), .ADDR_W(ADDR_W)
    ) conv1_inst (
        .clk(clk), .rst_n(rst_n), .start(conv1_start), .busy(conv1_busy), .done(conv1_done),
        .ifmap_addr(conv1_ifmap_addr), .ifmap_rdata(conv1_ifmap_rdata),
        .weight_addr(conv1_weight_addr), .weight_rdata(conv1_weight_rdata),
        .bias_addr(conv1_bias_addr), .bias_rdata(conv1_bias_rdata),
        .ofmap_we(conv1_ofmap_we), .ofmap_addr(conv1_ofmap_addr), .ofmap_wdata(conv1_ofmap_wdata)
    );

    // ------------------------------------------------------------------
    // pool1
    // ------------------------------------------------------------------
    reg  pool1_start;
    wire pool1_busy, pool1_done;
    wire [ADDR_W-1:0] pool1_ifmap_addr;
    reg  signed [DATA_W-1:0] pool1_ifmap_rdata;
    wire pool1_ofmap_we;
    wire [ADDR_W-1:0] pool1_ofmap_addr;
    wire signed [DATA_W-1:0] pool1_ofmap_wdata;

    always @(posedge clk) pool1_ifmap_rdata <= conv1_out_mem[pool1_ifmap_addr];
    always @(posedge clk) if (pool1_ofmap_we) pool1_out_mem[pool1_ofmap_addr] <= pool1_ofmap_wdata;

    maxpool #(
        .IMG_W(IMG0_W), .IMG_H(IMG0_H), .IN_CH(C1), .DATA_W(DATA_W), .ADDR_W(ADDR_W)
    ) pool1_inst (
        .clk(clk), .rst_n(rst_n), .start(pool1_start), .busy(pool1_busy), .done(pool1_done),
        .ifmap_addr(pool1_ifmap_addr), .ifmap_rdata(pool1_ifmap_rdata),
        .ofmap_we(pool1_ofmap_we), .ofmap_addr(pool1_ofmap_addr), .ofmap_wdata(pool1_ofmap_wdata)
    );

    // ------------------------------------------------------------------
    // conv2
    // ------------------------------------------------------------------
    reg  conv2_start;
    wire conv2_busy, conv2_done;
    wire [ADDR_W-1:0] conv2_ifmap_addr;
    reg  signed [DATA_W-1:0] conv2_ifmap_rdata;
    wire [ADDR_W-1:0] conv2_weight_addr;
    reg  signed [DATA_W-1:0] conv2_weight_rdata;
    wire [ADDR_W-1:0] conv2_bias_addr;
    reg  signed [BIAS_W-1:0] conv2_bias_rdata;
    wire conv2_ofmap_we;
    wire [ADDR_W-1:0] conv2_ofmap_addr;
    wire signed [DATA_W-1:0] conv2_ofmap_wdata;

    always @(posedge clk) conv2_ifmap_rdata  <= pool1_out_mem[conv2_ifmap_addr];
    always @(posedge clk) conv2_weight_rdata <= conv2_weight_mem[conv2_weight_addr];
    always @(posedge clk) conv2_bias_rdata   <= conv2_bias_mem[conv2_bias_addr];
    always @(posedge clk) if (conv2_ofmap_we) conv2_out_mem[conv2_ofmap_addr] <= conv2_ofmap_wdata;

    conv_layer #(
        .IMG_W(POOL1_W), .IMG_H(POOL1_H), .IN_CH(C1), .OUT_CH(C2), .KSIZE(3),
        .DATA_W(DATA_W), .BIAS_W(BIAS_W), .ACC_W(ACC_W), .FRAC_BITS(CONV2_FRAC_BITS), .ADDR_W(ADDR_W)
    ) conv2_inst (
        .clk(clk), .rst_n(rst_n), .start(conv2_start), .busy(conv2_busy), .done(conv2_done),
        .ifmap_addr(conv2_ifmap_addr), .ifmap_rdata(conv2_ifmap_rdata),
        .weight_addr(conv2_weight_addr), .weight_rdata(conv2_weight_rdata),
        .bias_addr(conv2_bias_addr), .bias_rdata(conv2_bias_rdata),
        .ofmap_we(conv2_ofmap_we), .ofmap_addr(conv2_ofmap_addr), .ofmap_wdata(conv2_ofmap_wdata)
    );

    // ------------------------------------------------------------------
    // pool2
    // ------------------------------------------------------------------
    reg  pool2_start;
    wire pool2_busy, pool2_done;
    wire [ADDR_W-1:0] pool2_ifmap_addr;
    reg  signed [DATA_W-1:0] pool2_ifmap_rdata;
    wire pool2_ofmap_we;
    wire [ADDR_W-1:0] pool2_ofmap_addr;
    wire signed [DATA_W-1:0] pool2_ofmap_wdata;

    always @(posedge clk) pool2_ifmap_rdata <= conv2_out_mem[pool2_ifmap_addr];
    always @(posedge clk) if (pool2_ofmap_we) pool2_out_mem[pool2_ofmap_addr] <= pool2_ofmap_wdata;

    maxpool #(
        .IMG_W(POOL1_W), .IMG_H(POOL1_H), .IN_CH(C2), .DATA_W(DATA_W), .ADDR_W(ADDR_W)
    ) pool2_inst (
        .clk(clk), .rst_n(rst_n), .start(pool2_start), .busy(pool2_busy), .done(pool2_done),
        .ifmap_addr(pool2_ifmap_addr), .ifmap_rdata(pool2_ifmap_rdata),
        .ofmap_we(pool2_ofmap_we), .ofmap_addr(pool2_ofmap_addr), .ofmap_wdata(pool2_ofmap_wdata)
    );

    // ------------------------------------------------------------------
    // conv3
    // ------------------------------------------------------------------
    reg  conv3_start;
    wire conv3_busy, conv3_done;
    wire [ADDR_W-1:0] conv3_ifmap_addr;
    reg  signed [DATA_W-1:0] conv3_ifmap_rdata;
    wire [ADDR_W-1:0] conv3_weight_addr;
    reg  signed [DATA_W-1:0] conv3_weight_rdata;
    wire [ADDR_W-1:0] conv3_bias_addr;
    reg  signed [BIAS_W-1:0] conv3_bias_rdata;
    wire conv3_ofmap_we;
    wire [ADDR_W-1:0] conv3_ofmap_addr;
    wire signed [DATA_W-1:0] conv3_ofmap_wdata;

    always @(posedge clk) conv3_ifmap_rdata  <= pool2_out_mem[conv3_ifmap_addr];
    always @(posedge clk) conv3_weight_rdata <= conv3_weight_mem[conv3_weight_addr];
    always @(posedge clk) conv3_bias_rdata   <= conv3_bias_mem[conv3_bias_addr];
    always @(posedge clk) if (conv3_ofmap_we) conv3_out_mem[conv3_ofmap_addr] <= conv3_ofmap_wdata;

    conv_layer #(
        .IMG_W(POOL2_W), .IMG_H(POOL2_H), .IN_CH(C2), .OUT_CH(C3), .KSIZE(3),
        .DATA_W(DATA_W), .BIAS_W(BIAS_W), .ACC_W(ACC_W), .FRAC_BITS(CONV3_FRAC_BITS), .ADDR_W(ADDR_W)
    ) conv3_inst (
        .clk(clk), .rst_n(rst_n), .start(conv3_start), .busy(conv3_busy), .done(conv3_done),
        .ifmap_addr(conv3_ifmap_addr), .ifmap_rdata(conv3_ifmap_rdata),
        .weight_addr(conv3_weight_addr), .weight_rdata(conv3_weight_rdata),
        .bias_addr(conv3_bias_addr), .bias_rdata(conv3_bias_rdata),
        .ofmap_we(conv3_ofmap_we), .ofmap_addr(conv3_ofmap_addr), .ofmap_wdata(conv3_ofmap_wdata)
    );

    // ------------------------------------------------------------------
    // pool3
    // ------------------------------------------------------------------
    reg  pool3_start;
    wire pool3_busy, pool3_done;
    wire [ADDR_W-1:0] pool3_ifmap_addr;
    reg  signed [DATA_W-1:0] pool3_ifmap_rdata;
    wire pool3_ofmap_we;
    wire [ADDR_W-1:0] pool3_ofmap_addr;
    wire signed [DATA_W-1:0] pool3_ofmap_wdata;

    always @(posedge clk) pool3_ifmap_rdata <= conv3_out_mem[pool3_ifmap_addr];
    always @(posedge clk) if (pool3_ofmap_we) pool3_out_mem[pool3_ofmap_addr] <= pool3_ofmap_wdata;

    maxpool #(
        .IMG_W(POOL2_W), .IMG_H(POOL2_H), .IN_CH(C3), .DATA_W(DATA_W), .ADDR_W(ADDR_W)
    ) pool3_inst (
        .clk(clk), .rst_n(rst_n), .start(pool3_start), .busy(pool3_busy), .done(pool3_done),
        .ifmap_addr(pool3_ifmap_addr), .ifmap_rdata(pool3_ifmap_rdata),
        .ofmap_we(pool3_ofmap_we), .ofmap_addr(pool3_ofmap_addr), .ofmap_wdata(pool3_ofmap_wdata)
    );

    // ------------------------------------------------------------------
    // gap
    // ------------------------------------------------------------------
    reg  gap_start;
    wire gap_busy, gap_done;
    wire [ADDR_W-1:0] gap_ifmap_addr;
    reg  signed [DATA_W-1:0] gap_ifmap_rdata;
    wire gap_ofmap_we;
    wire [ADDR_W-1:0] gap_ofmap_addr;
    wire signed [DATA_W-1:0] gap_ofmap_wdata;

    always @(posedge clk) gap_ifmap_rdata <= pool3_out_mem[gap_ifmap_addr];
    always @(posedge clk) if (gap_ofmap_we) gap_out_mem[gap_ofmap_addr] <= gap_ofmap_wdata;

    global_avg_pool #(
        .IMG_W(POOL3_W), .IMG_H(POOL3_H), .IN_CH(C3), .DATA_W(DATA_W), .ACC_W(ACC_W), .ADDR_W(ADDR_W)
    ) gap_inst (
        .clk(clk), .rst_n(rst_n), .start(gap_start), .busy(gap_busy), .done(gap_done),
        .ifmap_addr(gap_ifmap_addr), .ifmap_rdata(gap_ifmap_rdata),
        .ofmap_we(gap_ofmap_we), .ofmap_addr(gap_ofmap_addr), .ofmap_wdata(gap_ofmap_wdata)
    );

    // ------------------------------------------------------------------
    // fc
    // ------------------------------------------------------------------
    reg  fc_start;
    wire fc_busy, fc_done;
    wire [ADDR_W-1:0] fc_ifmap_addr;
    reg  signed [DATA_W-1:0] fc_ifmap_rdata;
    wire [ADDR_W-1:0] fc_weight_addr;
    reg  signed [DATA_W-1:0] fc_weight_rdata;
    wire [ADDR_W-1:0] fc_bias_addr;
    reg  signed [BIAS_W-1:0] fc_bias_rdata;
    wire fc_ofmap_we;
    wire [ADDR_W-1:0] fc_ofmap_addr;
    wire signed [SCORE_W-1:0] fc_ofmap_wdata;

    always @(posedge clk) fc_ifmap_rdata  <= gap_out_mem[fc_ifmap_addr];
    always @(posedge clk) fc_weight_rdata <= fc_weight_mem[fc_weight_addr];
    always @(posedge clk) fc_bias_rdata   <= fc_bias_mem[fc_bias_addr];
    always @(posedge clk) if (fc_ofmap_we) fc_out_mem[fc_ofmap_addr] <= fc_ofmap_wdata;

    fc #(
        .IN_FEATURES(C3), .OUT_FEATURES(NUM_CLASSES),
        .DATA_W(DATA_W), .BIAS_W(BIAS_W), .ACC_W(ACC_W), .FRAC_BITS(FC_FRAC_BITS),
        .SCORE_W(SCORE_W), .ADDR_W(ADDR_W)
    ) fc_inst (
        .clk(clk), .rst_n(rst_n), .start(fc_start), .busy(fc_busy), .done(fc_done),
        .ifmap_addr(fc_ifmap_addr), .ifmap_rdata(fc_ifmap_rdata),
        .weight_addr(fc_weight_addr), .weight_rdata(fc_weight_rdata),
        .bias_addr(fc_bias_addr), .bias_rdata(fc_bias_rdata),
        .ofmap_we(fc_ofmap_we), .ofmap_addr(fc_ofmap_addr), .ofmap_wdata(fc_ofmap_wdata)
    );

    // ------------------------------------------------------------------
    // argmax (inline - same FETCH/CAP-style 2-state read pattern as
    // maxpool's tap scan, extended to also track the winning index)
    // ------------------------------------------------------------------
    reg  [ADDR_W-1:0] argmax_addr;
    reg  signed [SCORE_W-1:0] argmax_rdata;
    always @(posedge clk) argmax_rdata <= fc_out_mem[argmax_addr];

    integer amx_idx;
    reg signed [SCORE_W-1:0] amx_best_val;
    integer amx_best_idx;
    integer amx_new_best_idx; // blocking scratch: the winning index AFTER considering this cycle's element (see S_ARGMAX_CAP)

    // ------------------------------------------------------------------
    // top-level FSM
    // ------------------------------------------------------------------
    localparam S_IDLE        = 5'd0,
               S_CONV1_START = 5'd1,  S_CONV1_WAIT = 5'd2,
               S_POOL1_START = 5'd3,  S_POOL1_WAIT = 5'd4,
               S_CONV2_START = 5'd5,  S_CONV2_WAIT = 5'd6,
               S_POOL2_START = 5'd7,  S_POOL2_WAIT = 5'd8,
               S_CONV3_START = 5'd9,  S_CONV3_WAIT = 5'd10,
               S_POOL3_START = 5'd11, S_POOL3_WAIT = 5'd12,
               S_GAP_START   = 5'd13, S_GAP_WAIT   = 5'd14,
               S_FC_START    = 5'd15, S_FC_WAIT    = 5'd16,
               S_ARGMAX_ADDR = 5'd17, S_ARGMAX_CAP = 5'd18,
               S_DONE        = 5'd19;

    reg [4:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            class_idx    <= 4'd0;
            conv1_start  <= 1'b0;
            pool1_start  <= 1'b0;
            conv2_start  <= 1'b0;
            pool2_start  <= 1'b0;
            conv3_start  <= 1'b0;
            pool3_start  <= 1'b0;
            gap_start    <= 1'b0;
            fc_start     <= 1'b0;
            argmax_addr  <= {ADDR_W{1'b0}};
            amx_idx      <= 0;
            amx_best_val <= {SCORE_W{1'b0}};
            amx_best_idx <= 0;
        end else begin
            // defaults: every stage's start is a 1-cycle pulse unless the
            // matching S_X_START branch below re-asserts it
            conv1_start <= 1'b0;
            pool1_start <= 1'b0;
            conv2_start <= 1'b0;
            pool2_start <= 1'b0;
            conv3_start <= 1'b0;
            pool3_start <= 1'b0;
            gap_start   <= 1'b0;
            fc_start    <= 1'b0;
            done        <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy  <= 1'b1;
                    state <= S_CONV1_START;
                end
            end

            // ---- conv1 --------------------------------------------------
            S_CONV1_START: begin conv1_start <= 1'b1; state <= S_CONV1_WAIT; end
            S_CONV1_WAIT:  begin if (conv1_done) state <= S_POOL1_START; end

            // ---- pool1 --------------------------------------------------
            S_POOL1_START: begin pool1_start <= 1'b1; state <= S_POOL1_WAIT; end
            S_POOL1_WAIT:  begin if (pool1_done) state <= S_CONV2_START; end

            // ---- conv2 --------------------------------------------------
            S_CONV2_START: begin conv2_start <= 1'b1; state <= S_CONV2_WAIT; end
            S_CONV2_WAIT:  begin if (conv2_done) state <= S_POOL2_START; end

            // ---- pool2 --------------------------------------------------
            S_POOL2_START: begin pool2_start <= 1'b1; state <= S_POOL2_WAIT; end
            S_POOL2_WAIT:  begin if (pool2_done) state <= S_CONV3_START; end

            // ---- conv3 --------------------------------------------------
            S_CONV3_START: begin conv3_start <= 1'b1; state <= S_CONV3_WAIT; end
            S_CONV3_WAIT:  begin if (conv3_done) state <= S_POOL3_START; end

            // ---- pool3 --------------------------------------------------
            S_POOL3_START: begin pool3_start <= 1'b1; state <= S_POOL3_WAIT; end
            S_POOL3_WAIT:  begin if (pool3_done) state <= S_GAP_START; end

            // ---- gap ----------------------------------------------------
            S_GAP_START: begin gap_start <= 1'b1; state <= S_GAP_WAIT; end
            S_GAP_WAIT:  begin if (gap_done) state <= S_FC_START; end

            // ---- fc -----------------------------------------------------
            S_FC_START: begin fc_start <= 1'b1; state <= S_FC_WAIT; end
            S_FC_WAIT: begin
                if (fc_done) begin
                    amx_idx     <= 0;
                    argmax_addr <= 0;
                    state       <= S_ARGMAX_ADDR;
                end
            end

            // ---- argmax ---------------------------------------------------
            S_ARGMAX_ADDR: begin
                // argmax_addr already stable (set on entry from S_FC_WAIT or
                // from the previous S_ARGMAX_CAP iteration)
                state <= S_ARGMAX_CAP;
            end

            S_ARGMAX_CAP: begin
                // argmax_rdata now corresponds to fc_out_mem[amx_idx]
                if (amx_idx == 0 || $signed(argmax_rdata) > $signed(amx_best_val)) begin
                    amx_best_val     <= argmax_rdata;
                    amx_new_best_idx = amx_idx;   // blocking: always the winner AFTER this element
                end else begin
                    amx_new_best_idx = amx_best_idx;
                end
                amx_best_idx <= amx_new_best_idx;

                if (amx_idx == NUM_CLASSES-1) begin
                    class_idx <= amx_new_best_idx[3:0]; // use the blocking value, not the stale non-blocking amx_best_idx
                    state     <= S_DONE;
                end else begin
                    amx_idx     <= amx_idx + 1;
                    argmax_addr <= amx_idx + 1;
                    state       <= S_ARGMAX_ADDR;
                end
            end

            // ---- done ---------------------------------------------------
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
