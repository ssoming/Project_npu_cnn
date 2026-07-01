`timescale 1ns / 1ps
// ============================================================================
// tb_fc - self-checking, category-based unit test
//   Pure Verilog-2001 (no SystemVerilog constructs: no '{...} array literals,
//   no `string` type, no %p) - same convention as the other testbenches.
//
//   Fixed shape: IN_FEATURES=5, OUT_FEATURES=3 (deliberately UNEQUAL, not
//   e.g. 4x4 - if the RTL's weight address ever accidentally used
//   inf*OUT_FEATURES+of instead of of*IN_FEATURES+inf, an unequal shape
//   makes that read from a completely different, wrong position instead of
//   "accidentally" landing on a valid-looking value by matrix symmetry).
//   All OUT_FEATURES=3 outputs are checked per case (not just one), which
//   matters here specifically because the concern is per-output-neuron
//   weight indexing.
//
//   Golden values were computed in Python using the exact RTL algorithm:
//   acc = bias[of] + sum_inf(ifmap[inf] * weight[of*IN_FEATURES+inf]),
//   then shifted = acc >> 7 (Python's >> is floor/arithmetic shift for
//   negative numbers too, matching Verilog's >>>). No ReLU, no saturation
//   (this is the final pre-argmax layer) - so an exact (diff==0) match is
//   expected for every output of every case.
//
//   Categories (9 cases total):
//     0: normal values
//     1: negative-mixed values
//     2: zero ifmap (bias-only path; distinct bias per output verifies
//        bias_addr=of is read correctly per output, not shared/stuck)
//     3: zero weight (bias-only via a different path than case 2)
//     4: large Q1.7-extreme values (+127/-128) with large bias - checks the
//        16-bit SCORE_W output does NOT clip/wrap (no saturation logic
//        exists in this module by design - see fc.v header)
//     5: indexing probe - weight[of][inf] = of*10+inf+1 (unique per (of,inf)
//        pair) with a single nonzero input at inf=2. This is the direct
//        check for the (out_features,in_features) address-order concern:
//        if of/inf were swapped in weight_index(), this reads completely
//        different weight values and every output would mismatch.
//     6-7: random
//     8: large negative bias dominates a small positive sum -> output must
//        be allowed to stay negative (no ReLU on this layer, unlike conv)
// ============================================================================
module tb_fc;

    localparam IN_FEATURES  = 5;
    localparam OUT_FEATURES = 3;
    localparam DATA_W    = 8;
    localparam BIAS_W    = 16;
    localparam ACC_W     = 32;
    localparam FRAC_BITS = 7;
    localparam SCORE_W   = 16;
    localparam ADDR_W    = 16;

    localparam NUM_CASES = 9;
    localparam NAME_W    = 28; // fixed-width (space-padded) case name, in characters

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
    wire signed [SCORE_W-1:0] ofmap_wdata;

    reg signed [DATA_W-1:0]  ifmap_mem  [0:IN_FEATURES-1];
    reg signed [DATA_W-1:0]  weight_mem [0:OUT_FEATURES*IN_FEATURES-1];
    reg signed [BIAS_W-1:0]  bias_mem   [0:OUT_FEATURES-1];
    reg signed [SCORE_W-1:0] ofmap_mem  [0:OUT_FEATURES-1];

    always @(posedge clk) ifmap_rdata  <= ifmap_mem[ifmap_addr];
    always @(posedge clk) weight_rdata <= weight_mem[weight_addr];
    always @(posedge clk) bias_rdata   <= bias_mem[bias_addr];
    always @(posedge clk) if (ofmap_we) ofmap_mem[ofmap_addr] <= ofmap_wdata;

    fc #(
        .IN_FEATURES(IN_FEATURES), .OUT_FEATURES(OUT_FEATURES),
        .DATA_W(DATA_W), .BIAS_W(BIAS_W), .ACC_W(ACC_W), .FRAC_BITS(FRAC_BITS),
        .SCORE_W(SCORE_W), .ADDR_W(ADDR_W)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
        .ifmap_addr(ifmap_addr), .ifmap_rdata(ifmap_rdata),
        .weight_addr(weight_addr), .weight_rdata(weight_rdata),
        .bias_addr(bias_addr), .bias_rdata(bias_rdata),
        .ofmap_we(ofmap_we), .ofmap_addr(ofmap_addr), .ofmap_wdata(ofmap_wdata)
    );

    always #5 clk = ~clk; // 100MHz

    // ------------------------------------------------------------------
    // test vectors - plain Verilog-2001 reg arrays, populated element-by-
    // element (no SystemVerilog '{...} literals, no array-typed task ports)
    // ------------------------------------------------------------------
    reg signed [DATA_W-1:0]  ifmap_case  [0:NUM_CASES-1][0:IN_FEATURES-1];
    reg signed [DATA_W-1:0]  weight_case [0:NUM_CASES-1][0:OUT_FEATURES*IN_FEATURES-1];
    reg signed [BIAS_W-1:0]  bias_case   [0:NUM_CASES-1][0:OUT_FEATURES-1];
    reg signed [SCORE_W-1:0] golden_case [0:NUM_CASES-1][0:OUT_FEATURES-1];
    reg [8*NAME_W-1:0]       case_name   [0:NUM_CASES-1];

    initial begin
        case_name[0] = "normal_values               ";
        ifmap_case[0][0] = 10;
        ifmap_case[0][1] = 20;
        ifmap_case[0][2] = 30;
        ifmap_case[0][3] = 15;
        ifmap_case[0][4] = 25;
        weight_case[0][0] = 5;
        weight_case[0][1] = 10;
        weight_case[0][2] = 15;
        weight_case[0][3] = 8;
        weight_case[0][4] = 12;
        weight_case[0][5] = 3;
        weight_case[0][6] = 7;
        weight_case[0][7] = 20;
        weight_case[0][8] = 4;
        weight_case[0][9] = 9;
        weight_case[0][10] = 6;
        weight_case[0][11] = 11;
        weight_case[0][12] = 2;
        weight_case[0][13] = 14;
        weight_case[0][14] = 8;
        bias_case[0][0] = 50;
        bias_case[0][1] = -20;
        bias_case[0][2] = 10;
        golden_case[0][0] = 9;
        golden_case[0][1] = 8;
        golden_case[0][2] = 5;

        case_name[1] = "negative_mixed              ";
        ifmap_case[1][0] = -10;
        ifmap_case[1][1] = 20;
        ifmap_case[1][2] = -30;
        ifmap_case[1][3] = 15;
        ifmap_case[1][4] = -5;
        weight_case[1][0] = 4;
        weight_case[1][1] = -6;
        weight_case[1][2] = 8;
        weight_case[1][3] = -2;
        weight_case[1][4] = 5;
        weight_case[1][5] = -3;
        weight_case[1][6] = 7;
        weight_case[1][7] = -9;
        weight_case[1][8] = 6;
        weight_case[1][9] = -4;
        weight_case[1][10] = 2;
        weight_case[1][11] = -8;
        weight_case[1][12] = 5;
        weight_case[1][13] = -7;
        weight_case[1][14] = 3;
        bias_case[1][0] = 0;
        bias_case[1][1] = 30;
        bias_case[1][2] = -15;
        golden_case[1][0] = -4;
        golden_case[1][1] = 4;
        golden_case[1][2] = -4;

        case_name[2] = "zero_ifmap                  ";
        ifmap_case[2][0] = 0;
        ifmap_case[2][1] = 0;
        ifmap_case[2][2] = 0;
        ifmap_case[2][3] = 0;
        ifmap_case[2][4] = 0;
        weight_case[2][0] = 9;
        weight_case[2][1] = -4;
        weight_case[2][2] = 3;
        weight_case[2][3] = 7;
        weight_case[2][4] = -2;
        weight_case[2][5] = 1;
        weight_case[2][6] = 5;
        weight_case[2][7] = -6;
        weight_case[2][8] = 8;
        weight_case[2][9] = 3;
        weight_case[2][10] = -7;
        weight_case[2][11] = 2;
        weight_case[2][12] = 4;
        weight_case[2][13] = -9;
        weight_case[2][14] = 6;
        bias_case[2][0] = 500;
        bias_case[2][1] = -300;
        bias_case[2][2] = 127;
        golden_case[2][0] = 3;
        golden_case[2][1] = -3;
        golden_case[2][2] = 0;

        case_name[3] = "zero_weight                 ";
        ifmap_case[3][0] = 12;
        ifmap_case[3][1] = -34;
        ifmap_case[3][2] = 56;
        ifmap_case[3][3] = -78;
        ifmap_case[3][4] = 90;
        weight_case[3][0] = 0;
        weight_case[3][1] = 0;
        weight_case[3][2] = 0;
        weight_case[3][3] = 0;
        weight_case[3][4] = 0;
        weight_case[3][5] = 0;
        weight_case[3][6] = 0;
        weight_case[3][7] = 0;
        weight_case[3][8] = 0;
        weight_case[3][9] = 0;
        weight_case[3][10] = 0;
        weight_case[3][11] = 0;
        weight_case[3][12] = 0;
        weight_case[3][13] = 0;
        weight_case[3][14] = 0;
        bias_case[3][0] = 1000;
        bias_case[3][1] = 0;
        bias_case[3][2] = -999;
        golden_case[3][0] = 7;
        golden_case[3][1] = 0;
        golden_case[3][2] = -8;

        case_name[4] = "large_values_no_saturate    ";
        ifmap_case[4][0] = 127;
        ifmap_case[4][1] = 127;
        ifmap_case[4][2] = -128;
        ifmap_case[4][3] = -128;
        ifmap_case[4][4] = 127;
        weight_case[4][0] = 127;
        weight_case[4][1] = 127;
        weight_case[4][2] = 127;
        weight_case[4][3] = 127;
        weight_case[4][4] = 127;
        weight_case[4][5] = -128;
        weight_case[4][6] = -128;
        weight_case[4][7] = -128;
        weight_case[4][8] = -128;
        weight_case[4][9] = -128;
        weight_case[4][10] = 127;
        weight_case[4][11] = -128;
        weight_case[4][12] = 127;
        weight_case[4][13] = -128;
        weight_case[4][14] = 127;
        bias_case[4][0] = 16000;
        bias_case[4][1] = -16000;
        bias_case[4][2] = 0;
        golden_case[4][0] = 249;
        golden_case[4][1] = -250;
        golden_case[4][2] = 126;

        case_name[5] = "indexing_probe_of_inf_order ";
        ifmap_case[5][0] = 0;
        ifmap_case[5][1] = 0;
        ifmap_case[5][2] = 50;
        ifmap_case[5][3] = 0;
        ifmap_case[5][4] = 0;
        weight_case[5][0] = 1;
        weight_case[5][1] = 2;
        weight_case[5][2] = 3;
        weight_case[5][3] = 4;
        weight_case[5][4] = 5;
        weight_case[5][5] = 11;
        weight_case[5][6] = 12;
        weight_case[5][7] = 13;
        weight_case[5][8] = 14;
        weight_case[5][9] = 15;
        weight_case[5][10] = 21;
        weight_case[5][11] = 22;
        weight_case[5][12] = 23;
        weight_case[5][13] = 24;
        weight_case[5][14] = 25;
        bias_case[5][0] = 0;
        bias_case[5][1] = 0;
        bias_case[5][2] = 0;
        golden_case[5][0] = 1;
        golden_case[5][1] = 5;
        golden_case[5][2] = 8;

        case_name[6] = "random_1                    ";
        ifmap_case[6][0] = 37;
        ifmap_case[6][1] = -51;
        ifmap_case[6][2] = 74;
        ifmap_case[6][3] = -104;
        ifmap_case[6][4] = -91;
        weight_case[6][0] = -80;
        weight_case[6][1] = 59;
        weight_case[6][2] = -99;
        weight_case[6][3] = -19;
        weight_case[6][4] = -109;
        weight_case[6][5] = -84;
        weight_case[6][6] = 94;
        weight_case[6][7] = 86;
        weight_case[6][8] = -93;
        weight_case[6][9] = -5;
        weight_case[6][10] = -82;
        weight_case[6][11] = 89;
        weight_case[6][12] = -98;
        weight_case[6][13] = -65;
        weight_case[6][14] = -14;
        bias_case[6][0] = 583;
        bias_case[6][1] = 569;
        bias_case[6][2] = 387;
        golden_case[6][0] = -7;
        golden_case[6][1] = 71;
        golden_case[6][2] = -51;

        case_name[7] = "random_2                    ";
        ifmap_case[7][0] = -97;
        ifmap_case[7][1] = 75;
        ifmap_case[7][2] = -103;
        ifmap_case[7][3] = -15;
        ifmap_case[7][4] = -105;
        weight_case[7][0] = -60;
        weight_case[7][1] = 20;
        weight_case[7][2] = 86;
        weight_case[7][3] = -55;
        weight_case[7][4] = -68;
        weight_case[7][5] = 29;
        weight_case[7][6] = -36;
        weight_case[7][7] = -76;
        weight_case[7][8] = -32;
        weight_case[7][9] = 62;
        weight_case[7][10] = -79;
        weight_case[7][11] = -96;
        weight_case[7][12] = -98;
        weight_case[7][13] = -23;
        weight_case[7][14] = 126;
        bias_case[7][0] = 786;
        bias_case[7][1] = 177;
        bias_case[7][2] = -249;
        golden_case[7][0] = 56;
        golden_case[7][1] = -28;
        golden_case[7][2] = -21;

        case_name[8] = "negative_bias_dominant      ";
        ifmap_case[8][0] = 5;
        ifmap_case[8][1] = 5;
        ifmap_case[8][2] = 5;
        ifmap_case[8][3] = 5;
        ifmap_case[8][4] = 5;
        weight_case[8][0] = 1;
        weight_case[8][1] = 1;
        weight_case[8][2] = 1;
        weight_case[8][3] = 1;
        weight_case[8][4] = 1;
        weight_case[8][5] = 1;
        weight_case[8][6] = 1;
        weight_case[8][7] = 1;
        weight_case[8][8] = 1;
        weight_case[8][9] = 1;
        weight_case[8][10] = 1;
        weight_case[8][11] = 1;
        weight_case[8][12] = 1;
        weight_case[8][13] = 1;
        weight_case[8][14] = 1;
        bias_case[8][0] = -5000;
        bias_case[8][1] = -5000;
        bias_case[8][2] = -5000;
        golden_case[8][0] = -39;
        golden_case[8][1] = -39;
        golden_case[8][2] = -39;
    end

    // ------------------------------------------------------------------
    // test driver
    // ------------------------------------------------------------------
    // display-only mask: golden_case/ofmap_mem are SCORE_W-bit regs, but
    // 'got_val'/'fail_got_vals' below are 32-bit 'integer's, which sign-
    // extend on assignment. Masking to SCORE_W bits (not a fixed pattern)
    // keeps the hex display consistent - same fix applied to the other
    // testbenches after the earlier 8bit-vs-16bit display mismatch.
    localparam [31:0] DISP_MASK = (1 << SCORE_W) - 1;

    integer t, o, i;
    integer pass_count;
    integer got_val;
    integer diff;
    integer mismatch_count;
    integer fail_got_vals [0:NUM_CASES-1];
    integer fail_count;
    reg     stuck;
    reg     case_ok;

    task run_case(input integer idx);
        begin
            for (i = 0; i < IN_FEATURES; i = i + 1) ifmap_mem[i] = ifmap_case[idx][i];
            for (i = 0; i < OUT_FEATURES*IN_FEATURES; i = i + 1) weight_mem[i] = weight_case[idx][i];
            for (i = 0; i < OUT_FEATURES; i = i + 1) bias_mem[i] = bias_case[idx][i];

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

            case_ok = 1'b1;
            mismatch_count = 0;
            for (o = 0; o < OUT_FEATURES; o = o + 1) begin
                if (ofmap_mem[o] !== golden_case[t][o]) begin
                    case_ok = 1'b0;
                    mismatch_count = mismatch_count + 1;
                end
            end

            if (case_ok) begin
                pass_count = pass_count + 1;
                $display("[TEST %0d] PASS  | %-30s | out0=0x%04x out1=0x%04x out2=0x%04x (all %0d outputs match)",
                          t+1, case_name[t],
                          golden_case[t][0] & DISP_MASK, golden_case[t][1] & DISP_MASK, golden_case[t][2] & DISP_MASK,
                          OUT_FEATURES);
            end else begin
                got_val = ofmap_mem[0]; // representative value for the stuck-at check below
                fail_got_vals[fail_count] = got_val;
                fail_count = fail_count + 1;

                $display("[TEST %0d] FAIL  | %-30s | %0d/%0d outputs mismatched",
                          t+1, case_name[t], mismatch_count, OUT_FEATURES);
                for (o = 0; o < OUT_FEATURES; o = o + 1) begin
                    got_val = ofmap_mem[o];
                    diff = got_val - golden_case[t][o];
                    if (diff != 0) begin
                        $display("           out[%0d]: expected=0x%04x, got=0x%04x (diff=%0d)",
                                  o, golden_case[t][o] & DISP_MASK, got_val & DISP_MASK, diff);
                    end
                end
                $display("           -> ReLU/saturate가 없는 순수 시프트 결과이므로 diff!=0은 전부 로직 버그(어드레싱/가중치 인덱싱/누산 오류)로 봐야 함");
                if (t == 5)
                    $display("           -> TEST 6은 (out_features,in_features) 가중치 인덱싱 순서 검증 케이스: 실패 시 weight_index()의 of/inf 순서를 최우선으로 의심할 것");
            end
        end

        $display("");
        if (pass_count == NUM_CASES)
            $display("=== SUMMARY: %0d / %0d PASS (100%%) ===", pass_count, NUM_CASES);
        else
            $display("=== SUMMARY: %0d / %0d PASS (%0d%%) ===", pass_count, NUM_CASES, (pass_count*100)/NUM_CASES);

        // stuck-at-fixed-value pattern check across failing cases (using each
        // failing case's output[0] as the representative sample)
        if (fail_count >= 2) begin
            stuck = 1'b1;
            for (i = 1; i < fail_count; i = i + 1)
                if (fail_got_vals[i] != fail_got_vals[0]) stuck = 1'b0;
            if (stuck)
                $display("!!! WARNING: %0d개 FAIL 케이스의 out[0] 값이 전부 0x%04x로 동일 - 특정 값/클래스 고정 버그 패턴(이전 프로젝트의 Loc 클래스 고정 버그와 유사) 의심 !!!",
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
