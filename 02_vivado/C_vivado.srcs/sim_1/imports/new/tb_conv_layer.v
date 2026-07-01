`timescale 1ns / 1ps
// ============================================================================
// tb_conv_layer - self-checking, category-based unit test
//   Pure Verilog-2001 (no SystemVerilog constructs: no '{...} array literals,
//   no `string` type, no %p) so it compiles correctly as a plain .v file
//   regardless of project file-type association.
//
//   One fixed conv_layer shape (IMG_W=IMG_H=3, IN_CH=2, OUT_CH=1, KSIZE=3) is
//   reused across 12 test cases; only the ifmap/weight/bias contents change
//   between cases. With a 3x3 image and "same" padding, the CENTER output
//   pixel (oy=1,ox=1) is the only position where all 9*IN_CH=18 taps are
//   real (non-padded) data, so it gives one clean, fully-exercised scalar
//   per case to compare.
//
//   Golden values were computed in Python/numpy using the *exact same*
//   integer fixed-point algorithm as the RTL (not a floating-point
//   approximation): acc(Q2.14) = bias + sum(ifmap*weight), ReLU, then a
//   plain truncating right-shift by 7 (valid because ReLU already forces
//   the value non-negative, so truncation == floor, with no separate
//   rounding step on either side). Because both sides use identical
//   integer arithmetic, the expected pass rate is 12/12 - any mismatch
//   indicates a real RTL bug, not quantization rounding. The diff-based
//   PASS/heuristic classification below is still implemented per spec in
//   case a mismatch does occur.
//
//   Categories (12 cases total):
//     1-2  : normal positive inputs
//     3-4  : negative-mixed inputs (ReLU clamp-to-zero check)
//     5-6  : zero-boundary (all-zero ifmap; all-zero weight + nonzero bias)
//     7-9  : Q1.7 extremes (+127/+127, -128/-128, +127/-128) - saturation
//            and negative*negative-is-positive sign-logic check
//     10-12: random (numpy RandomState(0), full int8/bias range)
// ============================================================================
module tb_conv_layer;

    localparam IMG_W     = 3;
    localparam IMG_H     = 3;
    localparam IN_CH     = 2;
    localparam OUT_CH    = 1;
    localparam KSIZE     = 3;
    localparam DATA_W    = 8;
    localparam BIAS_W    = 16;
    localparam ACC_W     = 32;
    localparam FRAC_BITS = 7;
    localparam ADDR_W    = 16;

    localparam IFMAP_N  = IN_CH*IMG_H*IMG_W;        // 18
    localparam WEIGHT_N = OUT_CH*IN_CH*KSIZE*KSIZE; // 18
    localparam OFMAP_N  = OUT_CH*IMG_H*IMG_W;       // 9
    localparam CENTER_IDX = (0*IMG_H + 1)*IMG_W + 1; // ofmap index of (oc=0,oy=1,ox=1) = 4

    localparam NUM_CASES = 12;
    localparam NAME_W    = 24; // fixed-width (space-padded) case name, in characters

    reg clk, rst_n, start;
    wire busy, done;

    wire [ADDR_W-1:0] ifmap_addr;
    reg  signed [DATA_W-1:0] ifmap_rdata;

    wire [ADDR_W-1:0] weight_addr;
    reg  signed [DATA_W-1:0] weight_rdata;

    wire [ADDR_W-1:0] bias_addr;
    reg  signed [BIAS_W-1:0] bias_rdata;

    wire ofmap_we;
    wire [ADDR_W-1:0] ofmap_addr;
    wire signed [DATA_W-1:0] ofmap_wdata;

    reg signed [DATA_W-1:0] ifmap_mem  [0:IFMAP_N-1];
    reg signed [DATA_W-1:0] weight_mem [0:WEIGHT_N-1];
    reg signed [BIAS_W-1:0] bias_mem   [0:0];
    reg signed [DATA_W-1:0] ofmap_mem  [0:OFMAP_N-1];

    always @(posedge clk) ifmap_rdata  <= ifmap_mem[ifmap_addr];
    always @(posedge clk) weight_rdata <= weight_mem[weight_addr];
    always @(posedge clk) bias_rdata   <= bias_mem[bias_addr];
    always @(posedge clk) if (ofmap_we) ofmap_mem[ofmap_addr] <= ofmap_wdata;

    conv_layer #(
        .IMG_W(IMG_W), .IMG_H(IMG_H), .IN_CH(IN_CH), .OUT_CH(OUT_CH), .KSIZE(KSIZE),
        .DATA_W(DATA_W), .BIAS_W(BIAS_W), .ACC_W(ACC_W), .FRAC_BITS(FRAC_BITS), .ADDR_W(ADDR_W)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
        .ifmap_addr(ifmap_addr), .ifmap_rdata(ifmap_rdata),
        .weight_addr(weight_addr), .weight_rdata(weight_rdata),
        .bias_addr(bias_addr), .bias_rdata(bias_rdata),
        .ofmap_we(ofmap_we), .ofmap_addr(ofmap_addr), .ofmap_wdata(ofmap_wdata)
    );

    always #5 clk = ~clk; // 100MHz

    // ------------------------------------------------------------------
    // test vectors (generated in Python/numpy - exact int fixed-point
    // golden values, not float approximations). Plain Verilog-2001
    // multi-dimensional reg arrays; populated element-by-element below
    // (no SystemVerilog '{...} literals).
    // ------------------------------------------------------------------
    reg signed [DATA_W-1:0] ifmap_case  [0:NUM_CASES-1][0:IFMAP_N-1];
    reg signed [DATA_W-1:0] weight_case [0:NUM_CASES-1][0:WEIGHT_N-1];
    reg signed [BIAS_W-1:0] bias_case   [0:NUM_CASES-1];
    reg signed [DATA_W-1:0] golden_case [0:NUM_CASES-1];
    reg [8*NAME_W-1:0]      case_name   [0:NUM_CASES-1]; // fixed-width ASCII, space-padded to NAME_W chars

    initial begin
        // ---- 1-2: normal positive ----
        case_name[0] = "normal_pos_1            ";
        ifmap_case[0][0]  = 49;
        ifmap_case[0][1]  = 52;
        ifmap_case[0][2]  = 58;
        ifmap_case[0][3]  = 5;
        ifmap_case[0][4]  = 8;
        ifmap_case[0][5]  = 8;
        ifmap_case[0][6]  = 44;
        ifmap_case[0][7]  = 14;
        ifmap_case[0][8]  = 24;
        ifmap_case[0][9]  = 26;
        ifmap_case[0][10]  = 55;
        ifmap_case[0][11]  = 41;
        ifmap_case[0][12]  = 28;
        ifmap_case[0][13]  = 11;
        ifmap_case[0][14]  = 29;
        ifmap_case[0][15]  = 29;
        ifmap_case[0][16]  = 17;
        ifmap_case[0][17]  = 6;
        weight_case[0][0] = 28;
        weight_case[0][1] = 29;
        weight_case[0][2] = 22;
        weight_case[0][3] = 30;
        weight_case[0][4] = 18;
        weight_case[0][5] = 13;
        weight_case[0][6] = 14;
        weight_case[0][7] = 25;
        weight_case[0][8] = 21;
        weight_case[0][9] = 10;
        weight_case[0][10] = 20;
        weight_case[0][11] = 5;
        weight_case[0][12] = 23;
        weight_case[0][13] = 29;
        weight_case[0][14] = 34;
        weight_case[0][15] = 24;
        weight_case[0][16] = 24;
        weight_case[0][17] = 19;
        bias_case[0] = 50;
        golden_case[0] = 84;

        case_name[1] = "normal_pos_2            ";
        ifmap_case[1][0]  = 8;
        ifmap_case[1][1]  = 1;
        ifmap_case[1][2]  = 2;
        ifmap_case[1][3]  = 10;
        ifmap_case[1][4]  = 26;
        ifmap_case[1][5]  = 1;
        ifmap_case[1][6]  = 11;
        ifmap_case[1][7]  = 21;
        ifmap_case[1][8]  = 24;
        ifmap_case[1][9]  = 4;
        ifmap_case[1][10]  = 12;
        ifmap_case[1][11]  = 19;
        ifmap_case[1][12]  = 24;
        ifmap_case[1][13]  = 29;
        ifmap_case[1][14]  = 3;
        ifmap_case[1][15]  = 1;
        ifmap_case[1][16]  = 1;
        ifmap_case[1][17]  = 5;
        weight_case[1][0] = 6;
        weight_case[1][1] = 7;
        weight_case[1][2] = 9;
        weight_case[1][3] = 18;
        weight_case[1][4] = 16;
        weight_case[1][5] = 5;
        weight_case[1][6] = 10;
        weight_case[1][7] = 11;
        weight_case[1][8] = 2;
        weight_case[1][9] = 2;
        weight_case[1][10] = 8;
        weight_case[1][11] = 10;
        weight_case[1][12] = 4;
        weight_case[1][13] = 7;
        weight_case[1][14] = 12;
        weight_case[1][15] = 15;
        weight_case[1][16] = 19;
        weight_case[1][17] = 1;
        bias_case[1] = 10;
        golden_case[1] = 13;

        // ---- 3-4: negative-mixed (ReLU check) ----
        case_name[2] = "neg_mixed_still_pos     ";
        ifmap_case[2][0]  = 40;
        ifmap_case[2][1]  = -10;
        ifmap_case[2][2]  = 20;
        ifmap_case[2][3]  = -5;
        ifmap_case[2][4]  = 30;
        ifmap_case[2][5]  = -15;
        ifmap_case[2][6]  = 25;
        ifmap_case[2][7]  = -20;
        ifmap_case[2][8]  = 10;
        ifmap_case[2][9]  = -8;
        ifmap_case[2][10]  = 15;
        ifmap_case[2][11]  = -12;
        ifmap_case[2][12]  = 22;
        ifmap_case[2][13]  = -30;
        ifmap_case[2][14]  = 18;
        ifmap_case[2][15]  = -5;
        ifmap_case[2][16]  = 10;
        ifmap_case[2][17]  = -9;
        weight_case[2][0] = 10;
        weight_case[2][1] = -5;
        weight_case[2][2] = 8;
        weight_case[2][3] = -6;
        weight_case[2][4] = 12;
        weight_case[2][5] = -4;
        weight_case[2][6] = 9;
        weight_case[2][7] = -7;
        weight_case[2][8] = 11;
        weight_case[2][9] = -3;
        weight_case[2][10] = 6;
        weight_case[2][11] = -9;
        weight_case[2][12] = 8;
        weight_case[2][13] = -11;
        weight_case[2][14] = 7;
        weight_case[2][15] = -4;
        weight_case[2][16] = 9;
        weight_case[2][17] = -6;
        bias_case[2] = 30;
        golden_case[2] = 20;

        case_name[3] = "neg_mixed_clip_zero     ";
        ifmap_case[3][0]  = -60;
        ifmap_case[3][1]  = -50;
        ifmap_case[3][2]  = -40;
        ifmap_case[3][3]  = -30;
        ifmap_case[3][4]  = -20;
        ifmap_case[3][5]  = -10;
        ifmap_case[3][6]  = -55;
        ifmap_case[3][7]  = -45;
        ifmap_case[3][8]  = -35;
        ifmap_case[3][9]  = -25;
        ifmap_case[3][10]  = -15;
        ifmap_case[3][11]  = -5;
        ifmap_case[3][12]  = -65;
        ifmap_case[3][13]  = -55;
        ifmap_case[3][14]  = -45;
        ifmap_case[3][15]  = -20;
        ifmap_case[3][16]  = -10;
        ifmap_case[3][17]  = -30;
        weight_case[3][0] = 40;
        weight_case[3][1] = 30;
        weight_case[3][2] = 20;
        weight_case[3][3] = 50;
        weight_case[3][4] = 10;
        weight_case[3][5] = 60;
        weight_case[3][6] = 25;
        weight_case[3][7] = 35;
        weight_case[3][8] = 15;
        weight_case[3][9] = 45;
        weight_case[3][10] = 5;
        weight_case[3][11] = 55;
        weight_case[3][12] = 30;
        weight_case[3][13] = 20;
        weight_case[3][14] = 40;
        weight_case[3][15] = 10;
        weight_case[3][16] = 50;
        weight_case[3][17] = 60;
        bias_case[3] = 0;
        golden_case[3] = 0;

        // ---- 5-6: zero boundary ----
        case_name[4] = "zero_ifmap              ";
        ifmap_case[4][0]  = 0;
        ifmap_case[4][1]  = 0;
        ifmap_case[4][2]  = 0;
        ifmap_case[4][3]  = 0;
        ifmap_case[4][4]  = 0;
        ifmap_case[4][5]  = 0;
        ifmap_case[4][6]  = 0;
        ifmap_case[4][7]  = 0;
        ifmap_case[4][8]  = 0;
        ifmap_case[4][9]  = 0;
        ifmap_case[4][10]  = 0;
        ifmap_case[4][11]  = 0;
        ifmap_case[4][12]  = 0;
        ifmap_case[4][13]  = 0;
        ifmap_case[4][14]  = 0;
        ifmap_case[4][15]  = 0;
        ifmap_case[4][16]  = 0;
        ifmap_case[4][17]  = 0;
        weight_case[4][0] = 42;
        weight_case[4][1] = -1;
        weight_case[4][2] = -47;
        weight_case[4][3] = 40;
        weight_case[4][4] = 21;
        weight_case[4][5] = 70;
        weight_case[4][6] = -16;
        weight_case[4][7] = -32;
        weight_case[4][8] = -94;
        weight_case[4][9] = 96;
        weight_case[4][10] = -53;
        weight_case[4][11] = 27;
        weight_case[4][12] = 31;
        weight_case[4][13] = 0;
        weight_case[4][14] = 80;
        weight_case[4][15] = -22;
        weight_case[4][16] = 43;
        weight_case[4][17] = 48;
        bias_case[4] = 0;
        golden_case[4] = 0;

        case_name[5] = "zero_weight_bias200     ";
        ifmap_case[5][0]  = 86;
        ifmap_case[5][1]  = -77;
        ifmap_case[5][2]  = 41;
        ifmap_case[5][3]  = 17;
        ifmap_case[5][4]  = -15;
        ifmap_case[5][5]  = -52;
        ifmap_case[5][6]  = -51;
        ifmap_case[5][7]  = -31;
        ifmap_case[5][8]  = 69;
        ifmap_case[5][9]  = 63;
        ifmap_case[5][10]  = 92;
        ifmap_case[5][11]  = -5;
        ifmap_case[5][12]  = 97;
        ifmap_case[5][13]  = -6;
        ifmap_case[5][14]  = -100;
        ifmap_case[5][15]  = 13;
        ifmap_case[5][16]  = 78;
        ifmap_case[5][17]  = -64;
        weight_case[5][0] = 0;
        weight_case[5][1] = 0;
        weight_case[5][2] = 0;
        weight_case[5][3] = 0;
        weight_case[5][4] = 0;
        weight_case[5][5] = 0;
        weight_case[5][6] = 0;
        weight_case[5][7] = 0;
        weight_case[5][8] = 0;
        weight_case[5][9] = 0;
        weight_case[5][10] = 0;
        weight_case[5][11] = 0;
        weight_case[5][12] = 0;
        weight_case[5][13] = 0;
        weight_case[5][14] = 0;
        weight_case[5][15] = 0;
        weight_case[5][16] = 0;
        weight_case[5][17] = 0;
        bias_case[5] = 200;
        golden_case[5] = 1;

        // ---- 7-9: Q1.7 boundary / saturation ----
        case_name[6] = "max_pos_x_max_pos_satlow";
        ifmap_case[6][0]  = 127;
        ifmap_case[6][1]  = 127;
        ifmap_case[6][2]  = 127;
        ifmap_case[6][3]  = 127;
        ifmap_case[6][4]  = 127;
        ifmap_case[6][5]  = 127;
        ifmap_case[6][6]  = 127;
        ifmap_case[6][7]  = 127;
        ifmap_case[6][8]  = 127;
        ifmap_case[6][9]  = 127;
        ifmap_case[6][10]  = 127;
        ifmap_case[6][11]  = 127;
        ifmap_case[6][12]  = 127;
        ifmap_case[6][13]  = 127;
        ifmap_case[6][14]  = 127;
        ifmap_case[6][15]  = 127;
        ifmap_case[6][16]  = 127;
        ifmap_case[6][17]  = 127;
        weight_case[6][0] = 127;
        weight_case[6][1] = 127;
        weight_case[6][2] = 127;
        weight_case[6][3] = 127;
        weight_case[6][4] = 127;
        weight_case[6][5] = 127;
        weight_case[6][6] = 127;
        weight_case[6][7] = 127;
        weight_case[6][8] = 127;
        weight_case[6][9] = 127;
        weight_case[6][10] = 127;
        weight_case[6][11] = 127;
        weight_case[6][12] = 127;
        weight_case[6][13] = 127;
        weight_case[6][14] = 127;
        weight_case[6][15] = 127;
        weight_case[6][16] = 127;
        weight_case[6][17] = 127;
        bias_case[6] = 0;
        golden_case[6] = 127;

        case_name[7] = "max_neg_x_max_neg_satlow";
        ifmap_case[7][0]  = -128;
        ifmap_case[7][1]  = -128;
        ifmap_case[7][2]  = -128;
        ifmap_case[7][3]  = -128;
        ifmap_case[7][4]  = -128;
        ifmap_case[7][5]  = -128;
        ifmap_case[7][6]  = -128;
        ifmap_case[7][7]  = -128;
        ifmap_case[7][8]  = -128;
        ifmap_case[7][9]  = -128;
        ifmap_case[7][10]  = -128;
        ifmap_case[7][11]  = -128;
        ifmap_case[7][12]  = -128;
        ifmap_case[7][13]  = -128;
        ifmap_case[7][14]  = -128;
        ifmap_case[7][15]  = -128;
        ifmap_case[7][16]  = -128;
        ifmap_case[7][17]  = -128;
        weight_case[7][0] = -128;
        weight_case[7][1] = -128;
        weight_case[7][2] = -128;
        weight_case[7][3] = -128;
        weight_case[7][4] = -128;
        weight_case[7][5] = -128;
        weight_case[7][6] = -128;
        weight_case[7][7] = -128;
        weight_case[7][8] = -128;
        weight_case[7][9] = -128;
        weight_case[7][10] = -128;
        weight_case[7][11] = -128;
        weight_case[7][12] = -128;
        weight_case[7][13] = -128;
        weight_case[7][14] = -128;
        weight_case[7][15] = -128;
        weight_case[7][16] = -128;
        weight_case[7][17] = -128;
        bias_case[7] = 0;
        golden_case[7] = 127;

        case_name[8] = "max_pos_x_max_neg_clip0 ";
        ifmap_case[8][0]  = 127;
        ifmap_case[8][1]  = 127;
        ifmap_case[8][2]  = 127;
        ifmap_case[8][3]  = 127;
        ifmap_case[8][4]  = 127;
        ifmap_case[8][5]  = 127;
        ifmap_case[8][6]  = 127;
        ifmap_case[8][7]  = 127;
        ifmap_case[8][8]  = 127;
        ifmap_case[8][9]  = 127;
        ifmap_case[8][10]  = 127;
        ifmap_case[8][11]  = 127;
        ifmap_case[8][12]  = 127;
        ifmap_case[8][13]  = 127;
        ifmap_case[8][14]  = 127;
        ifmap_case[8][15]  = 127;
        ifmap_case[8][16]  = 127;
        ifmap_case[8][17]  = 127;
        weight_case[8][0] = -128;
        weight_case[8][1] = -128;
        weight_case[8][2] = -128;
        weight_case[8][3] = -128;
        weight_case[8][4] = -128;
        weight_case[8][5] = -128;
        weight_case[8][6] = -128;
        weight_case[8][7] = -128;
        weight_case[8][8] = -128;
        weight_case[8][9] = -128;
        weight_case[8][10] = -128;
        weight_case[8][11] = -128;
        weight_case[8][12] = -128;
        weight_case[8][13] = -128;
        weight_case[8][14] = -128;
        weight_case[8][15] = -128;
        weight_case[8][16] = -128;
        weight_case[8][17] = -128;
        bias_case[8] = 0;
        golden_case[8] = 0;

        // ---- 10-12: random (numpy RandomState(0)) ----
        case_name[9] = "random_1                ";
        ifmap_case[9][0]  = 34;
        ifmap_case[9][1]  = -80;
        ifmap_case[9][2]  = -35;
        ifmap_case[9][3]  = 3;
        ifmap_case[9][4]  = -30;
        ifmap_case[9][5]  = -86;
        ifmap_case[9][6]  = 77;
        ifmap_case[9][7]  = -16;
        ifmap_case[9][8]  = 103;
        ifmap_case[9][9]  = 21;
        ifmap_case[9][10]  = 73;
        ifmap_case[9][11]  = -1;
        ifmap_case[9][12]  = -128;
        ifmap_case[9][13]  = 10;
        ifmap_case[9][14]  = -14;
        ifmap_case[9][15]  = -85;
        ifmap_case[9][16]  = 58;
        ifmap_case[9][17]  = -1;
        weight_case[9][0] = -105;
        weight_case[9][1] = 59;
        weight_case[9][2] = 2;
        weight_case[9][3] = -7;
        weight_case[9][4] = -30;
        weight_case[9][5] = -66;
        weight_case[9][6] = 35;
        weight_case[9][7] = 94;
        weight_case[9][8] = -5;
        weight_case[9][9] = 67;
        weight_case[9][10] = -46;
        weight_case[9][11] = 46;
        weight_case[9][12] = 99;
        weight_case[9][13] = 20;
        weight_case[9][14] = 81;
        weight_case[9][15] = -78;
        weight_case[9][16] = 27;
        weight_case[9][17] = -114;
        bias_case[9] = 1113;
        golden_case[9] = 0;

        case_name[10] = "random_2                ";
        ifmap_case[10][0]  = -70;
        ifmap_case[10][1]  = 65;
        ifmap_case[10][2]  = -92;
        ifmap_case[10][3]  = -118;
        ifmap_case[10][4]  = -42;
        ifmap_case[10][5]  = -85;
        ifmap_case[10][6]  = -24;
        ifmap_case[10][7]  = -117;
        ifmap_case[10][8]  = -126;
        ifmap_case[10][9]  = -77;
        ifmap_case[10][10]  = -48;
        ifmap_case[10][11]  = -96;
        ifmap_case[10][12]  = 54;
        ifmap_case[10][13]  = 0;
        ifmap_case[10][14]  = -90;
        ifmap_case[10][15]  = -109;
        ifmap_case[10][16]  = 46;
        ifmap_case[10][17]  = -86;
        weight_case[10][0] = -13;
        weight_case[10][1] = 56;
        weight_case[10][2] = 60;
        weight_case[10][3] = 104;
        weight_case[10][4] = -51;
        weight_case[10][5] = -98;
        weight_case[10][6] = -104;
        weight_case[10][7] = -3;
        weight_case[10][8] = -126;
        weight_case[10][9] = -125;
        weight_case[10][10] = -34;
        weight_case[10][11] = 98;
        weight_case[10][12] = -21;
        weight_case[10][13] = -115;
        weight_case[10][14] = -16;
        weight_case[10][15] = -88;
        weight_case[10][16] = -56;
        weight_case[10][17] = -109;
        bias_case[10] = -1393;
        golden_case[10] = 127;

        case_name[11] = "random_3                ";
        ifmap_case[11][0]  = -56;
        ifmap_case[11][1]  = 26;
        ifmap_case[11][2]  = 66;
        ifmap_case[11][3]  = 120;
        ifmap_case[11][4]  = 52;
        ifmap_case[11][5]  = -61;
        ifmap_case[11][6]  = 108;
        ifmap_case[11][7]  = -67;
        ifmap_case[11][8]  = -114;
        ifmap_case[11][9]  = -32;
        ifmap_case[11][10]  = -124;
        ifmap_case[11][11]  = 67;
        ifmap_case[11][12]  = 109;
        ifmap_case[11][13]  = 11;
        ifmap_case[11][14]  = 124;
        ifmap_case[11][15]  = -42;
        ifmap_case[11][16]  = 77;
        ifmap_case[11][17]  = -7;
        weight_case[11][0] = -19;
        weight_case[11][1] = -53;
        weight_case[11][2] = 56;
        weight_case[11][3] = -112;
        weight_case[11][4] = 24;
        weight_case[11][5] = 29;
        weight_case[11][6] = 21;
        weight_case[11][7] = -18;
        weight_case[11][8] = -103;
        weight_case[11][9] = 80;
        weight_case[11][10] = 60;
        weight_case[11][11] = -7;
        weight_case[11][12] = -10;
        weight_case[11][13] = -11;
        weight_case[11][14] = 61;
        weight_case[11][15] = -45;
        weight_case[11][16] = 33;
        weight_case[11][17] = -24;
        bias_case[11] = -1072;
        golden_case[11] = 31;
    end

    // ------------------------------------------------------------------
    // test driver
    // ------------------------------------------------------------------
    // display-only mask: golden_case/ofmap_mem are DATA_W-bit regs, but
    // 'got'/'fail_got_vals' below are 32-bit 'integer's. Assigning a signed
    // DATA_W-bit value into a wider 'integer' sign-extends it, so masking
    // with a fixed DISP_MASK would show extra 'f' padding for negative
    // values instead of the true DATA_W-bit pattern (harmless here since
    // conv_layer's outputs are always ReLU'd, i.e. non-negative, but fixed
    // for consistency with tb_maxpool.v).
    localparam [31:0] DISP_MASK = (1 << DATA_W) - 1;

    integer t, i;
    integer pass_count;
    integer got;
    integer diff;
    integer fail_got_vals [0:NUM_CASES-1];
    integer fail_count;
    reg     stuck;

    task run_case(input integer idx);
        begin
            for (i = 0; i < IFMAP_N; i = i + 1)  ifmap_mem[i]  = ifmap_case[idx][i];
            for (i = 0; i < WEIGHT_N; i = i + 1) weight_mem[i] = weight_case[idx][i];
            bias_mem[0] = bias_case[idx];

            rst_n = 0;
            repeat (3) @(posedge clk);
            rst_n = 1;
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;

            wait (done);
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 0; rst_n = 0; start = 0;
        pass_count = 0;
        fail_count = 0;

        #1; // ensure the case-data initial block above has settled (same time-0 region)

        for (t = 0; t < NUM_CASES; t = t + 1) begin
            run_case(t);

            got  = ofmap_mem[CENTER_IDX];
            diff = got - golden_case[t];

            if (diff == 0) begin
                pass_count = pass_count + 1;
                $display("[TEST %0d] PASS  | %-26s | expected=0x%04x, got=0x%04x",
                          t+1, case_name[t], golden_case[t] & DISP_MASK, got & DISP_MASK);
            end else begin
                fail_got_vals[fail_count] = got;
                fail_count = fail_count + 1;
                $display("[TEST %0d] FAIL  | %-26s | expected=0x%04x, got=0x%04x (diff=%0d)",
                          t+1, case_name[t], golden_case[t] & DISP_MASK, got & DISP_MASK, diff);
                $write("           ifmap_case[%0d]  = ", t);
                for (i = 0; i < IFMAP_N; i = i + 1) $write("%0d ", ifmap_case[t][i]);
                $display("");
                $write("           weight_case[%0d] = ", t);
                for (i = 0; i < WEIGHT_N; i = i + 1) $write("%0d ", weight_case[t][i]);
                $display("");
                $display("           bias_case[%0d]   = %0d", t, bias_case[t]);
                if (diff >= -2 && diff <= 2)
                    $display("           -> diff<=2: 양자화 반올림 오차로 추정 (단, 이 테스트는 golden도 truncating shift라 원래 diff=0이어야 함 - 재확인 필요)");
                else
                    $display("           -> diff 큼/부호 다름: 로직 버그 가능성 높음 (어드레싱/누산 오류 의심)");
            end
        end

        $display("");
        if (pass_count == NUM_CASES)
            $display("=== SUMMARY: %0d / %0d PASS (100%%) ===", pass_count, NUM_CASES);
        else
            $display("=== SUMMARY: %0d / %0d PASS (%0d%%) ===", pass_count, NUM_CASES, (pass_count*100)/NUM_CASES);

        // stuck-at-fixed-value pattern check across failing cases
        if (fail_count >= 2) begin
            stuck = 1'b1;
            for (i = 1; i < fail_count; i = i + 1)
                if (fail_got_vals[i] != fail_got_vals[0]) stuck = 1'b0;
            if (stuck)
                $display("!!! WARNING: %0d개 FAIL 케이스의 got 값이 전부 0x%04x로 동일 - 특정 값 고정 버그 패턴(이전 class-4 고정 버그와 유사) 의심 !!!",
                          fail_count, fail_got_vals[0] & DISP_MASK);
        end

        $finish;
    end

    initial begin
        #5_000_000;
        $display("=== TIMEOUT: simulation did not finish in time ===");
        $finish;
    end

endmodule
