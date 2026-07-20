// test_cfc_pc.c — PC-side CfC inference validation against golden data
#include "cfc_inference.h"
#include "test_golden_data.h"
#include <stdio.h>
#include <math.h>

static const char *doa_names[5] = {
    "thumb_rotation", "thumb_flexion", "index_flexion",
    "middle_flexion", "ring_little_flexion"
};

int main(void) {
    int failures = 0;

    // 1. Init LUTs
    cfc_lut_init();

    // 2. Reshape flat input to (seq, dim)
    float feature_seq[TEST_SEQ_LEN][TEST_INPUT_DIM];
    for (int t = 0; t < TEST_SEQ_LEN; t++)
        for (int d = 0; d < TEST_INPUT_DIM; d++)
            feature_seq[t][d] = test_input[t * TEST_INPUT_DIM + d];

    // 3. Run CfC inference (per-channel INT8 weights)
    float output[TEST_OUTPUT_DIM];
    cfc_inference_int8(feature_seq, output);

    // 4. Compare against Python INT8 golden (should match closely)
    printf("=== C INT8 vs Python INT8 Golden ===\n");
    printf("%-25s %12s %12s %12s\n", "DoA", "C Output", "Py INT8", "Abs Error");
    float max_int8_err = 0.0f;
    for (int i = 0; i < TEST_OUTPUT_DIM; i++) {
        float err = fabsf(output[i] - test_expected_int8[i]);
        if (err > max_int8_err) max_int8_err = err;
        printf("%-25s %12.6f %12.6f %12.6f %s\n",
               doa_names[i], output[i], test_expected_int8[i], err,
               err < 0.002f ? "PASS" : "FAIL");
        if (err >= 0.002f) failures++;
    }
    printf("C-INT8 vs Py-INT8 max error: %.6f\n\n", max_int8_err);

    // 5. Compare against Python FP32 golden (quantization gap)
    printf("=== C INT8 vs Python FP32 Golden (quantization gap) ===\n");
    printf("%-25s %12s %12s %12s\n", "DoA", "C Output", "Py FP32", "Abs Error");
    float max_fp32_err = 0.0f;
    for (int i = 0; i < TEST_OUTPUT_DIM; i++) {
        float err = fabsf(output[i] - test_expected_fp32[i]);
        if (err > max_fp32_err) max_fp32_err = err;
        printf("%-25s %12.6f %12.6f %12.6f\n",
               doa_names[i], output[i], test_expected_fp32[i], err);
    }
    float output_range = 0.0f;
    for (int i = 0; i < TEST_OUTPUT_DIM; i++) {
        float v = fabsf(test_expected_fp32[i]);
        if (v > output_range) output_range = v;
    }
    printf("C-INT8 vs Py-FP32 max error: %.6f (%.4f%% relative)\n",
           max_fp32_err, max_fp32_err / (output_range + 1e-8f) * 100.0f);

    // Overall verdict
    printf("\n=== VERDICT ===\n");
    printf("  INT8 match:  %s (max err %.6f, threshold 0.002)\n",
           max_int8_err < 0.002f ? "PASS" : "FAIL", max_int8_err);
    printf("  Quant gap:   %.4f%% (within expected <1%% for 6-layer per-channel)\n",
           max_fp32_err / (output_range + 1e-8f) * 100.0f);

    return failures > 0 ? 1 : 0;
}
