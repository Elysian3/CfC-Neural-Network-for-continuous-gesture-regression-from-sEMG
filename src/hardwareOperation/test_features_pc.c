#include <math.h>
#include <stdio.h>

#include "../../firmware/main/weights.h"
#include "../../firmware/main/cfc_inference.h"
#include "../../firmware/main/features.h"
#include "../../firmware/main/normalization.h"

#define TEST_SAMPLES 500
#define TEST_WINDOWS 2

int main(void) {
    static float filtered[TEST_SAMPLES * N_CHANNELS];
    float features[TEST_WINDOWS * N_CHANNELS];
    for (int sample = 0; sample < TEST_SAMPLES; sample++) {
        for (int channel = 0; channel < N_CHANNELS; channel++) {
            filtered[sample * N_CHANNELS + channel] = (float)(channel + 1);
        }
    }

    const int windows = extract_rms_features(
        filtered,
        TEST_SAMPLES,
        N_CHANNELS,
        features,
        TEST_WINDOWS
    );
    if (windows != TEST_WINDOWS) {
        fprintf(stderr, "expected %d RMS windows, got %d\n", TEST_WINDOWS, windows);
        return 1;
    }
    for (int window = 0; window < TEST_WINDOWS; window++) {
        for (int channel = 0; channel < N_CHANNELS; channel++) {
            const float expected = (float)(channel + 1);
            const float actual = features[window * N_CHANNELS + channel];
            if (fabsf(actual - expected) > 1e-5f) {
                fprintf(stderr, "RMS mismatch at window=%d channel=%d\n", window, channel);
                return 2;
            }
        }
    }

    float normalized[CFC_INPUT_DIM];
    for (int i = 0; i < CFC_INPUT_DIM; i++) {
        normalized[i] = features[i];
    }
    mulaw_encode(normalized, CFC_INPUT_DIM);
    for (int i = 0; i < CFC_INPUT_DIM; i++) {
        const float raw = (float)(i + 1);
        const float expected = log1pf(MU_LAW_MU * raw) / log1pf(MU_LAW_MU);
        if (fabsf(normalized[i] - expected) > 1e-6f) {
            fprintf(stderr, "feature mu-law mismatch at %d\n", i);
            return 3;
        }
    }

    cfc_lut_init();
    cfc_reset_state();
    float output[CFC_OUTPUT_DIM];
    cfc_single_step(normalized, output);
    for (int i = 0; i < CFC_OUTPUT_DIM; i++) {
        if (!isfinite(output[i])) {
            fprintf(stderr, "non-finite integrated output at %d\n", i);
            return 4;
        }
    }

    puts("Feature-to-CfC PC golden test passed");
    return 0;
}
