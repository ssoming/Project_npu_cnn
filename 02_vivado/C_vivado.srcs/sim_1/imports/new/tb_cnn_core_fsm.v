`timescale 1ns / 1ps
// ============================================================================
// tb_cnn_core_fsm - REAL 48x48 wafer image test (real trained weights)
//   Pure Verilog-2001 (no SystemVerilog constructs), same convention as
//   every other testbench in this project.
//
//   This supersedes the earlier 16x16 smoke test (which already proved the
//   8-stage wiring/handshake sequencing is correct - 1/1 PASS). This test
//   uses cnn_core_fsm's DEFAULT parameters, i.e. the real model size
//   (48x48x1 -> 8 -> 16 -> 64 -> 9 classes) and the real trained+quantized
//   weight .mem files at /home/kimyujeong/PycharmProjects/wafer_NPU/python/
//   python/mem/. Nine real WM-811K wafer images were selected from X.npy/
//   y.npy, one per class (Center, Donut, Edge-Loc, Edge-Ring, Loc,
//   Near-full, Random, Scratch, none) - deliberately mixing the classes the
//   float model was strong on (none, Edge-Ring) with ones it was weaker on
//   (Scratch, Loc, Donut), specifically to surface any bias toward a single
//   class the way the previous project's integration bug did.
//
//   *** IMPORTANT - what "golden" means here and why ***
//   There are two different reference values per case, and they measure
//   two different things:
//     - quant_pred_idx[t]: the class predicted by a Python pipeline that
//       replicates the RTL's exact int8/int16 fixed-point algorithm
//       (documented in each module's header, already unit-tested to exact
//       (diff==0) agreement in isolation) run against the SAME quantized
//       weights the RTL loads (parsed back from the .mem files, not
//       re-derived). This is the PASS/FAIL criterion below, because it is
//       the only reference that isolates "did the RTL implement the
//       quantized algorithm correctly" from "how much did int8
//       quantization itself degrade accuracy".
//     - float_pred_idx[t]: the original float Keras model's prediction
//       (task04_model_test.py-style, no quantization at all) - printed for
//       reference only, NOT used for PASS/FAIL.
//
//   *** class-collapse bug and its fix (history, for context) ***
//   An earlier version of this test found the quantized pipeline collapsing
//   onto 2-3 classes regardless of input (Edge-Loc/Edge-Ring/none only, on a
//   180-image stratified check: 23.3% accuracy, 0 predictions for 6 of the
//   9 classes). Fixing fc weight clipping (task05_export_weights.py now
//   picks a clipping-free frac_bits per layer's weights: conv1=6,
//   conv2/conv3=7, fc=3) did NOT fix this. Ablating weight-quantization vs
//   activation-quantization noise separately (weights-only-quantized: 74%
//   acc, matches float; activations-only-quantized: 22% acc, same collapse
//   pattern) isolated the real cause to ACTIVATION quantization range, not
//   weights: activations were quantized as Q1.7 (representable range
//   0..0.9921875), copying the input image's own range, but real
//   intermediate activations (post-ReLU conv1/2/3 outputs, GAP output) were
//   measured (model.predict, N=2000) to reach up to ~16.0, so 12-18% of
//   ALL activation values were silently saturating to 0.992 at every single
//   layer. The fix (in task05_export_weights.py) keeps DATA_W=8 UNCHANGED
//   and just lowers the shared activation ACT_FRAC_BITS from 7 to 2
//   (representable range 0..31.75, ~2x margin over the measured max) -
//   trading fractional resolution for integer range on the same 8 bits.
//   Confirmed on a 72-image stratified sample: accuracy 23.3%->63.9%
//   (float baseline on the same sample: 77.8%), agreement 33.9%->68.1%,
//   max single-class prediction share 51.1%->26.4% (well under the 70%
//   collapse threshold). No RTL port width or FRAC_BITS parameter changed -
//   conv_layer.v/maxpool.v/global_avg_pool.v/fc.v are agnostic to what
//   ACT_FRAC_BITS the surrounding numbers mean (they only ever see raw
//   8/16-bit integers), so only the .mem file contents (bias values, image
//   quantization) changed, plus the quant_pred_idx golden values below.
//
//   Stuck-at check: separately from the quant_pred_idx comparison, this
//   test also checks whether the 9 RTL outputs show any diversity at all
//   (previous project's bug was ALL predictions collapsing to one fixed
//   class regardless of input). After the fix above, quant_pred_idx below
//   already spans 5 distinct classes across the 9 cases, so a stuck-at
//   result here would indicate a NEW regression, not the old known issue.
//
//   NOTE: at real 48x48/8/16/64 size, this runs approximately 4.4M cycles
//   per image x 9 images =~ 40M cycles (~400ms simulated time at 100MHz).
//   This will take a while in xsim - that is expected, not a hang.
// ============================================================================
module tb_cnn_core_fsm;

    localparam DATA_W = 8;
    localparam ADDR_W = 16;
    localparam IMG0_W = 48;
    localparam IMG0_H = 48;
    localparam IMG_N  = IMG0_W*IMG0_H;
    localparam NUM_CASES = 9;

    reg clk, rst_n, start;
    wire busy, done;
    wire [3:0] class_idx;

    reg                      img_we;
    reg  [ADDR_W-1:0]        img_addr;
    reg  signed [DATA_W-1:0] img_wdata;

    // default parameters == real model (48x48x1 -> C1=8 -> C2=16 -> C3=64
    // -> 9 classes) and default file paths already point at the real
    // trained weight .mem files - no overrides needed.
    cnn_core_fsm dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done), .class_idx(class_idx),
        .img_we(img_we), .img_addr(img_addr), .img_wdata(img_wdata)
    );

    always #5 clk = ~clk; // 100MHz

    // ------------------------------------------------------------------
    // real wafer images (one per class), int8-quantized at ACT_FRAC_BITS=2
    // (see quant_meta.json), selected from X.npy/y.npy - see conversation
    // for the selection script
    // ------------------------------------------------------------------
    localparam MEM_DIR = "/home/kimyujeong/PycharmProjects/wafer_NPU/python/python/mem/test_images/";

    reg signed [DATA_W-1:0] image0_mem [0:IMG_N-1];
    reg signed [DATA_W-1:0] image1_mem [0:IMG_N-1];
    reg signed [DATA_W-1:0] image2_mem [0:IMG_N-1];
    reg signed [DATA_W-1:0] image3_mem [0:IMG_N-1];
    reg signed [DATA_W-1:0] image4_mem [0:IMG_N-1];
    reg signed [DATA_W-1:0] image5_mem [0:IMG_N-1];
    reg signed [DATA_W-1:0] image6_mem [0:IMG_N-1];
    reg signed [DATA_W-1:0] image7_mem [0:IMG_N-1];
    reg signed [DATA_W-1:0] image8_mem [0:IMG_N-1];

    initial $readmemh({MEM_DIR, "image_0_Center.mem"},    image0_mem);
    initial $readmemh({MEM_DIR, "image_1_Donut.mem"},     image1_mem);
    initial $readmemh({MEM_DIR, "image_2_Edge-Loc.mem"},  image2_mem);
    initial $readmemh({MEM_DIR, "image_3_Edge-Ring.mem"}, image3_mem);
    initial $readmemh({MEM_DIR, "image_4_Loc.mem"},       image4_mem);
    initial $readmemh({MEM_DIR, "image_5_Near-full.mem"}, image5_mem);
    initial $readmemh({MEM_DIR, "image_6_Random.mem"},    image6_mem);
    initial $readmemh({MEM_DIR, "image_7_Scratch.mem"},   image7_mem);
    initial $readmemh({MEM_DIR, "image_8_none.mem"},      image8_mem);

    // class index <-> name mapping (LabelEncoder alphabetical order, same
    // as task03_train_gap.py / task05_export_weights.py)
    reg [8*16-1:0] class_name [0:8];
    initial begin
        class_name[0] = "Center          ";
        class_name[1] = "Donut           ";
        class_name[2] = "Edge-Loc        ";
        class_name[3] = "Edge-Ring       ";
        class_name[4] = "Loc             ";
        class_name[5] = "Near-full       ";
        class_name[6] = "Random          ";
        class_name[7] = "Scratch         ";
        class_name[8] = "none            ";
    end

    // per-case reference data (see manifest.json alongside the image .mem files)
    integer true_class_idx  [0:NUM_CASES-1];
    integer float_pred_idx  [0:NUM_CASES-1]; // task04-style float model prediction - reference only
    integer quant_pred_idx  [0:NUM_CASES-1]; // PASS/FAIL criterion - RTL must match this
    initial begin
        // quant_pred_idx recomputed after the activation-range fix
        // (ACT_FRAC_BITS 7->2, DATA_W unchanged; weight frac_bits unchanged:
        // conv1=6, conv2/conv3=7, fc=3 - see quant_meta.json / manifest.json)
        true_class_idx[0] = 0; float_pred_idx[0] = 0; quant_pred_idx[0] = 0; // Center
        true_class_idx[1] = 1; float_pred_idx[1] = 1; quant_pred_idx[1] = 1; // Donut
        true_class_idx[2] = 2; float_pred_idx[2] = 2; quant_pred_idx[2] = 4; // Edge-Loc (quant predicts Loc)
        true_class_idx[3] = 3; float_pred_idx[3] = 3; quant_pred_idx[3] = 8; // Edge-Ring (quant predicts none)
        true_class_idx[4] = 4; float_pred_idx[4] = 4; quant_pred_idx[4] = 4; // Loc
        true_class_idx[5] = 5; float_pred_idx[5] = 5; quant_pred_idx[5] = 6; // Near-full (quant predicts Random)
        true_class_idx[6] = 6; float_pred_idx[6] = 6; quant_pred_idx[6] = 0; // Random (quant predicts Center)
        true_class_idx[7] = 7; float_pred_idx[7] = 2; quant_pred_idx[7] = 8; // Scratch (float model itself also misclassifies this one, as Edge-Loc)
        true_class_idx[8] = 8; float_pred_idx[8] = 8; quant_pred_idx[8] = 8; // none
    end

    // ------------------------------------------------------------------
    // test driver
    // ------------------------------------------------------------------
    integer t, p;
    integer pass_count;
    integer got_vals [0:NUM_CASES-1];
    reg     stuck;

    task run_case(input integer case_idx);
        begin
            for (p = 0; p < IMG_N; p = p + 1) begin
                @(posedge clk);
                img_we   = 1;
                img_addr = p;
                case (case_idx)
                    0: img_wdata = image0_mem[p];
                    1: img_wdata = image1_mem[p];
                    2: img_wdata = image2_mem[p];
                    3: img_wdata = image3_mem[p];
                    4: img_wdata = image4_mem[p];
                    5: img_wdata = image5_mem[p];
                    6: img_wdata = image6_mem[p];
                    7: img_wdata = image7_mem[p];
                    8: img_wdata = image8_mem[p];
                    default: img_wdata = 0;
                endcase
            end
            @(posedge clk);
            img_we = 0;

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
        img_we = 0; img_addr = 0; img_wdata = 0;
        pass_count = 0;

        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        for (t = 0; t < NUM_CASES; t = t + 1) begin
            run_case(t);
            got_vals[t] = class_idx;

            if (class_idx === quant_pred_idx[t]) begin
                pass_count = pass_count + 1;
                $display("[TEST %0d] PASS  | img_class=%-16s (true=%0d) | RTL=%0d(%-16s) golden(quant)=%0d(%-16s) | float_model_ref=%0d(%-16s)",
                          t+1, class_name[t], true_class_idx[t],
                          class_idx, class_name[class_idx],
                          quant_pred_idx[t], class_name[quant_pred_idx[t]],
                          float_pred_idx[t], class_name[float_pred_idx[t]]);
            end else begin
                $display("[TEST %0d] FAIL  | img_class=%-16s (true=%0d) | RTL=%0d(%-16s) golden(quant)=%0d(%-16s) | float_model_ref=%0d(%-16s)",
                          t+1, class_name[t], true_class_idx[t],
                          class_idx, class_name[class_idx],
                          quant_pred_idx[t], class_name[quant_pred_idx[t]],
                          float_pred_idx[t], class_name[float_pred_idx[t]]);
                $display("           fc scores (RTL, via dut.fc_out_mem):");
                for (p = 0; p < 9; p = p + 1)
                    $display("             fc_out_mem[%0d] (%s) = %0d", p, class_name[p], dut.fc_out_mem[p]);
                $display("           -> RTL != golden(quant): 8단계 중 어디서 어긋났는지 dut.gap_out_mem / dut.pool3_out_mem 등 계층 참조로 좁혀갈 것 (float_model_ref와의 불일치는 양자화 문제이지 이 FAIL의 원인이 아님)");
            end
        end

        $display("");
        if (pass_count == NUM_CASES)
            $display("=== SUMMARY (vs quantized golden): %0d / %0d PASS (100%%) ===", pass_count, NUM_CASES);
        else
            $display("=== SUMMARY (vs quantized golden): %0d / %0d PASS (%0d%%) ===", pass_count, NUM_CASES, (pass_count*100)/NUM_CASES);

        // stuck-at check: do ALL 9 RTL outputs collapse to a single class
        // regardless of input? (the previous project's failure mode)
        stuck = 1'b1;
        for (t = 1; t < NUM_CASES; t = t + 1)
            if (got_vals[t] != got_vals[0]) stuck = 1'b0;
        if (stuck)
            $display("!!! WARNING: 9개 이미지 전부 RTL class_idx=%0d(%-16s)로 동일 - 입력과 무관하게 고정되는 버그 패턴(이전 프로젝트와 동일 증상) 강하게 의심 !!!",
                      got_vals[0], class_name[got_vals[0]]);
        else
            $display("stuck-at check: RTL 출력이 %0d개 이미지에서 서로 다른 클래스를 냄 (입력 무관 고정 버그는 아님)",
                      NUM_CASES);

        $display("");
        $display("(참고) float 모델 vs quantized golden 일치도 4/9 (이 9장 기준) - activation 쏠림 버그 수정(ACT_FRAC_BITS 7->2) 후 72장 표본에서는 accuracy 23.3%%->63.9%%, 최대 단일클래스 쏠림 51.1%%->26.4%%로 개선 확인됨. 남은 차이는 int8 PTQ의 정상적인 정확도 손실 수준이며 RTL PASS/FAIL과는 별개의 이슈입니다.");

        $finish;
    end

    initial begin
        #800_000_000; // ~4.4M cycles/image x 9 images at real 48x48/8/16/64 size - expected to take a while
        $display("=== TIMEOUT: simulation did not finish in time ===");
        $finish;
    end

endmodule
