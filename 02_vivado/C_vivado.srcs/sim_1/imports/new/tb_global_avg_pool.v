`timescale 1ns / 1ps
// ============================================================================
// tb_global_avg_pool - self-checking, category-based unit test
//   Pure Verilog-2001 (no SystemVerilog constructs: no '{...} array literals,
//   no `string` type, no %p) - same convention as tb_conv_layer.v/tb_maxpool.v.
//
//   One fixed global_avg_pool shape (IMG_W=IMG_H=3, IN_CH=1, N_PIX=9) is
//   reused across 10 test cases; only the ifmap contents (all 9 pixels of
//   the single channel) change between cases. The checked output is
//   ofmap index 0 (= channel 0's average).
//
//   Golden values were computed in Python replicating the RTL's exact
//   round-half-away-from-zero algorithm: avg = sign(acc) * round(|acc|/N)
//   via (|acc| + N/2) / N with truncating integer division - NOT a
//   floating-point mean - so an exact (diff==0) match is expected for
//   every case; any mismatch is a real RTL bug (wrong sum, wrong address,
//   or a rounding/truncation discrepancy), not a rounding "tolerance".
//   N_PIX=9 is intentionally not a power of two, so almost every case
//   exercises a non-exact division; cases 4/5 specifically sit at a
//   remainder (5/9 and its negative) that DISTINGUISHES rounding from
//   plain truncation, so a truncate-only bug would be caught here.
//
//   Categories (10 cases total):
//     0: normal positive average
//     1: negative-mixed average
//     2: all-zero input
//     3: all-identical input (exact division, no rounding needed - sanity check)
//     4: remainder=5/9 -> rounds UP (would FAIL under plain truncation)
//     5: same as 4 but negative sum -> rounds to -8, not -7 (symmetric rounding check)
//     6: all inputs at +127 (DATA_W max) -> average must equal 127 exactly
//     7: all inputs at -128 (DATA_W min) -> average must equal -128 exactly
//     8: mixed +127/-128/0 -> sign cancellation check
//     9: remainder=4/9 -> rounds DOWN (matches floor, sanity check for the other direction)
// ============================================================================
module tb_global_avg_pool;

    localparam IMG_W  = 3;
    localparam IMG_H  = 3;
    localparam IN_CH  = 1;
    localparam DATA_W = 8;
    localparam ACC_W  = 32;
    localparam ADDR_W = 16;

    localparam IFMAP_N = IN_CH*IMG_H*IMG_W; // 9
    localparam OFMAP_N = IN_CH;             // 1
    localparam CHECK_IDX = 0;               // channel 0's averaged output

    localparam NUM_CASES = 10;
    localparam NAME_W    = 24; // fixed-width (space-padded) case name, in characters

    reg clk, rst_n, start;
    wire busy, done;

    wire [ADDR_W-1:0] ifmap_addr;
    reg  signed [DATA_W-1:0] ifmap_rdata;

    wire ofmap_we;
    wire [ADDR_W-1:0] ofmap_addr;
    wire signed [DATA_W-1:0] ofmap_wdata;

    reg signed [DATA_W-1:0] ifmap_mem [0:IFMAP_N-1];
    reg signed [DATA_W-1:0] ofmap_mem [0:OFMAP_N-1];

    always @(posedge clk) ifmap_rdata <= ifmap_mem[ifmap_addr];
    always @(posedge clk) if (ofmap_we) ofmap_mem[ofmap_addr] <= ofmap_wdata;

    global_avg_pool #(
        .IMG_W(IMG_W), .IMG_H(IMG_H), .IN_CH(IN_CH), .DATA_W(DATA_W), .ACC_W(ACC_W), .ADDR_W(ADDR_W)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
        .ifmap_addr(ifmap_addr), .ifmap_rdata(ifmap_rdata),
        .ofmap_we(ofmap_we), .ofmap_addr(ofmap_addr), .ofmap_wdata(ofmap_wdata)
    );

    always #5 clk = ~clk; // 100MHz

    // ------------------------------------------------------------------
    // test vectors - plain Verilog-2001 reg arrays, populated via a task
    // (no SystemVerilog '{...} literals)
    // ------------------------------------------------------------------
    reg signed [DATA_W-1:0] ifmap_case  [0:NUM_CASES-1][0:IFMAP_N-1];
    reg signed [DATA_W-1:0] golden_case [0:NUM_CASES-1];
    reg [8*NAME_W-1:0]      case_name   [0:NUM_CASES-1];

    task init_case;
        input integer idx;
        input signed [DATA_W-1:0] v0, v1, v2, v3, v4, v5, v6, v7, v8;
        input signed [DATA_W-1:0] golden;
        input [8*NAME_W-1:0] nm;
        begin
            ifmap_case[idx][0] = v0;
            ifmap_case[idx][1] = v1;
            ifmap_case[idx][2] = v2;
            ifmap_case[idx][3] = v3;
            ifmap_case[idx][4] = v4;
            ifmap_case[idx][5] = v5;
            ifmap_case[idx][6] = v6;
            ifmap_case[idx][7] = v7;
            ifmap_case[idx][8] = v8;
            golden_case[idx] = golden;
            case_name[idx]   = nm;
        end
    endtask

    initial begin
        // sum=200 -> 200/9=22.22 -> round 22
        init_case(0, 10,20,30, 15,25,35, 5,40,20,   22, "normal_positive         ");
        // sum=-13 -> -13/9=-1.44 -> round -1
        init_case(1, -40,20,-10, 30,-5,15, -25,10,-8, -1, "negative_mixed          ");
        // sum=0 -> 0
        init_case(2,   0,0,0,     0,0,0,    0,0,0,     0, "all_zero                ");
        // sum=450 -> exact, no rounding
        init_case(3,  50,50,50,  50,50,50, 50,50,50,   50, "all_same_value          ");
        // sum=68 -> 68/9=7.555 -> rounds UP to 8 (truncation would wrongly give 7)
        init_case(4,  10,10,10,  10,10,10,  4, 2, 2,    8, "remainder_round_up      ");
        // sum=-68 -> rounds to -8 (naive toward-zero truncation would wrongly give -7)
        init_case(5, -10,-10,-10,-10,-10,-10,-4,-2,-2,  -8, "remainder_round_up_neg  ");
        // all +127 -> average must be exactly 127 (DATA_W max, no overflow)
        init_case(6, 127,127,127,127,127,127,127,127,127,127, "extreme_all_max         ");
        // all -128 -> average must be exactly -128 (DATA_W min, no overflow)
        init_case(7,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128, "extreme_all_min          ");
        // sum=-4 -> -4/9=-0.444 -> rounds to 0
        init_case(8, 127,-128,127,-128,127,-128,127,-128,0,   0, "extreme_mixed_sign      ");
        // sum=40 -> 40/9=4.444 -> rounds DOWN to 4 (matches floor - sanity check)
        init_case(9,  10,10,10,   4, 2, 2,  1, 1, 0,    4, "small_remainder_down    ");
    end

    // ------------------------------------------------------------------
    // test driver
    // ------------------------------------------------------------------
    // display-only mask: golden_case/ofmap_mem are DATA_W-bit regs, but
    // 'got'/'fail_got_vals' below are 32-bit 'integer's, which sign-extend
    // on assignment. Masking to DATA_W bits (not a fixed 16'hffff) keeps
    // the hex display consistent regardless of source width - this is the
    // same fix applied to tb_conv_layer.v/tb_maxpool.v after the earlier
    // 8bit-vs-16bit display mismatch.
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
            for (i = 0; i < IFMAP_N; i = i + 1) ifmap_mem[i] = ifmap_case[idx][i];

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

            got  = ofmap_mem[CHECK_IDX];
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
                $display("           ifmap(9 values) = %0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d",
                          ifmap_case[t][0], ifmap_case[t][1], ifmap_case[t][2], ifmap_case[t][3],
                          ifmap_case[t][4], ifmap_case[t][5], ifmap_case[t][6], ifmap_case[t][7], ifmap_case[t][8]);
                $display("           -> divide/round이 없는 순수 truncation 결과와 golden이 다르면 반올림 로직 버그, 그 외 큰 diff는 합산/어드레싱 버그로 봐야 함");
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
                $display("!!! WARNING: %0d개 FAIL 케이스의 got 값이 전부 0x%04x로 동일 - 특정 값 고정 버그 패턴 의심 !!!",
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
