`timescale 1ns / 1ps
// ============================================================================
// tb_maxpool - self-checking, category-based unit test
//   Pure Verilog-2001 (no SystemVerilog constructs: no '{...} array literals,
//   no `string` type, no %p) - same lesson learned from tb_conv_layer.v.
//
//   One fixed maxpool shape (IMG_W=IMG_H=4, IN_CH=1) is reused across 10
//   test cases; only the ifmap contents change between cases. Each case
//   only sets the top-left 2x2 window (ifmap indices 0,1,4,5 = (iy,ix) =
//   (0,0),(0,1),(1,0),(1,1)); the rest of the 4x4 image is filled with 0
//   (unchecked - it only affects the other 3 output pixels, which this
//   test does not evaluate). The checked output is ofmap index 0, i.e.
//   (ic=0,oy=0,ox=0), the max of that top-left window.
//
//   Since max-pooling is an exact integer operation (no shift/rounding
//   step anywhere), the golden value must match the RTL output exactly -
//   any nonzero diff is a logic bug, not quantization rounding.
//
//   Categories (10 cases total), each also pins down which of the 4 taps
//   holds the max, to catch tap-address/init/overwrite bugs:
//     0: normal, max at top-right tap
//     1: normal, max at bottom-right tap (last tap - tests final compare)
//     2: all negative (signed-compare check)
//     3: mixed sign
//     4: all zero
//     5: 4 identical values (no ambiguity, tests tie handling)
//     6: max at top-left tap (first/init tap - tests init isn't overwritten)
//     7: extreme DATA_W boundary values (-128 / 127 mixed)
//     8: all negative, max at last tap
//     9: max at bottom-left tap
// ============================================================================
module tb_maxpool;

    localparam IMG_W  = 4;
    localparam IMG_H  = 4;
    localparam IN_CH  = 1;
    localparam DATA_W = 8;
    localparam ADDR_W = 16;

    localparam OUT_W = IMG_W / 2;
    localparam OUT_H = IMG_H / 2;

    localparam IFMAP_N = IN_CH*IMG_H*IMG_W; // 16
    localparam OFMAP_N = IN_CH*OUT_H*OUT_W; // 4
    localparam CHECK_IDX = (0*OUT_H + 0)*OUT_W + 0; // ofmap index of (ic=0,oy=0,ox=0) = 0

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

    maxpool #(
        .IMG_W(IMG_W), .IMG_H(IMG_H), .IN_CH(IN_CH), .DATA_W(DATA_W), .ADDR_W(ADDR_W)
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
        input signed [DATA_W-1:0] v_tl;
        input signed [DATA_W-1:0] v_tr;
        input signed [DATA_W-1:0] v_bl;
        input signed [DATA_W-1:0] v_br;
        input signed [DATA_W-1:0] golden;
        input [8*NAME_W-1:0] nm;
        integer k;
        begin
            for (k = 0; k < IFMAP_N; k = k + 1) ifmap_case[idx][k] = 0;
            ifmap_case[idx][0] = v_tl; // (iy=0,ix=0) top-left
            ifmap_case[idx][1] = v_tr; // (iy=0,ix=1) top-right
            ifmap_case[idx][4] = v_bl; // (iy=1,ix=0) bottom-left
            ifmap_case[idx][5] = v_br; // (iy=1,ix=1) bottom-right
            golden_case[idx] = golden;
            case_name[idx]   = nm;
        end
    endtask

    initial begin
        init_case(0,  10,  50,  30,  20,  50, "normal_max_top_right    ");
        init_case(1,   5,  10,  15,  90,  90, "normal_max_bottom_right ");
        init_case(2, -50, -10, -30,  -5,  -5, "all_negative            ");
        init_case(3, -20,  15,  -5,   8,  15, "mixed_sign              ");
        init_case(4,   0,   0,   0,   0,   0, "all_zero                ");
        init_case(5,  42,  42,  42,  42,  42, "tie_identical           ");
        init_case(6,  99,   1,   2,   3,  99, "max_top_left            ");
        init_case(7,-128, 127,  -1,   0, 127, "extreme_boundary        ");
        init_case(8,-100, -90, -80, -70, -70, "negative_max_last       ");
        init_case(9,   7,   3,  88,  -4,  88, "max_bottom_left         ");
    end

    // ------------------------------------------------------------------
    // test driver
    // ------------------------------------------------------------------
    // display-only mask: golden_case/ofmap_mem are DATA_W-bit regs, but
    // 'got'/'fail_got_vals' below are 32-bit 'integer's. Assigning a signed
    // DATA_W-bit value into a wider 'integer' sign-extends it (e.g. 8'hFB
    // -> 32'hFFFFFFFB), so masking with a fixed DISP_MASK shows extra 'f'
    // padding for negative values instead of the true DATA_W-bit pattern.
    // Masking to DATA_W bits instead keeps the hex display consistent
    // regardless of which variable's width it came from.
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
                $display("           window(tl,tr,bl,br) = %0d, %0d, %0d, %0d",
                          ifmap_case[t][0], ifmap_case[t][1], ifmap_case[t][4], ifmap_case[t][5]);
                $display("           -> maxpool은 shift/rounding이 없는 순수 정수 비교이므로 diff!=0은 전부 로직 버그(어드레싱/비교 오류)로 봐야 함");
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
