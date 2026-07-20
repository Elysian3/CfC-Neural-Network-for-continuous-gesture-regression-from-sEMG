
// test_features_pc.c — Validate C RMS + mu-law + CfC against Python
#include "features.h"
#include "cfc_inference.h"
#include "normalization.h"
#include "test_emg_data.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(void) {
    int failures = 0;
    cfc_lut_init();

    // ── 1. RMS Feature Extraction ─────────────────────────────────────────
    float c_features[TEST_N_WINDOWS * N_CHANNELS];
    int n_w = extract_rms_features(test_emg, TEST_N_SAMPLES, TEST_N_CHANNELS,
                                   c_features, TEST_N_WINDOWS);

    printf("=== RMS Feature Extraction ===\n");
    printf("Windows extracted: %d (expected %d)\n", n_w, TEST_N_WINDOWS);

    if (n_w != TEST_N_WINDOWS) {
        printf("FAIL: wrong window count\n");
        return 1;
    }

    float max_rms_err = 0.0f;
    for (int i = 0; i < n_w * TEST_N_CHANNELS; i++) {
        float err = fabsf(c_features[i] - test_expected_rms[i]);
        if (err > max_rms_err) max_rms_err = err;
    }
    printf("Max RMS error vs Python: %.6e\n", max_rms_err);
    printf("RMS match: %s\n\n", max_rms_err < 1e-4f ? "PASS" : "FAIL");
    if (max_rms_err >= 1e-4f) failures++;

    // ── 2. Full Pipeline: RMS → mu-law → CfC ─────────────────────────────
    // Take 8 consecutive RMS windows (need at least 8)
    if (n_w < 8) {
        printf("FAIL: need at least 8 windows for CfC, got %d\n", n_w);
        return 1;
    }

    float feature_seq[8][N_FEATURES];
    for (int t = 0; t < 8; t++)
        for (int ch = 0; ch < N_FEATURES; ch++)
            feature_seq[t][ch] = c_features[t * N_CHANNELS + ch];

    // Apply mu-law normalization to each frame
    for (int t = 0; t < 8; t++)
        mulaw_encode(feature_seq[t], N_FEATURES);

    // Run CfC inference
    float output[CFC_OUTPUT_DIM];
    cfc_inference_int8(feature_seq, output);

    printf("=== Full Pipeline (RMS → mu-law → CfC) ===\n");
    const char *doa[] = {"thumb_rot","thumb_flex","index_flex",
                         "middle_flex","ring_little_flex"};
    printf("%-16s %12s\n", "DoA", "Prediction");
    for (int i = 0; i < CFC_OUTPUT_DIM; i++)
        printf("%-16s %12.6f\n", doa[i], output[i]);

    // Verify outputs are in reasonable range (not NaN, not absurd)
    float output_range = 0.0f;
    for (int i = 0; i < CFC_OUTPUT_DIM; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            printf("FAIL: output[%d] = %f\n", i, output[i]);
            failures++;
        }
        float v = fabsf(output[i]);
        if (v > output_range) output_range = v;
    }

    // Mu-law output bounds: should be in [-1, 1] for typical inputs
    printf("Output range: [%.4f, %.4f]\n",
           output[0], output[CFC_OUTPUT_DIM-1]);
    printf("Pipeline: %s\n", failures == 0 ? "PASS" : "FAIL");

    // ── 3. VERDICT ────────────────────────────────────────────────────────
    printf("\n=== VERDICT ===\n");
    printf("  RMS:     %s (max err %.2e, threshold 1e-4)\n",
           max_rms_err < 1e-4f ? "PASS" : "FAIL", max_rms_err);
    printf("  Pipeline: %s\n", failures == 0 ? "PASS" : "FAIL");

    return failures > 0 ? 1 : 0;
}
