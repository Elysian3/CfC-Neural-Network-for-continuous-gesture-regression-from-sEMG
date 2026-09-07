#include <math.h>
#include <stdio.h>

#include "../../firmware/main/weights.h"
#include "../../firmware/main/cfc_inference.h"

static int close_enough(float left, float right, float tolerance) {
    return fabsf(left - right) <= tolerance;
}

int main(void) {
    float encoded[3] = {-0.5f, 0.0f, 0.5f};
    const float center[3] = {10.0f, 20.0f, 30.0f};
    const float scale[3] = {2.0f, 3.0f, 4.0f};
    const float expected[3] = {9.88235294f, 20.0f, 30.23529412f};
    cfc_inverse_mulaw(encoded, 3, center, scale, 255.0f);
    for (int i = 0; i < 3; i++) {
        if (!close_enough(encoded[i], expected[i], 1e-5f)) {
            fprintf(stderr, "inverse mu-law mismatch at %d: %.8f\n", i, encoded[i]);
            return 1;
        }
    }

    float sequence[CFC_SEQ_LEN][CFC_INPUT_DIM];
    for (int t = 0; t < CFC_SEQ_LEN; t++) {
        for (int i = 0; i < CFC_INPUT_DIM; i++) {
            sequence[t][i] = 0.001f * (float)((t + 1) * (i + 1));
        }
    }

    cfc_lut_init();
    float sequence_output[CFC_OUTPUT_DIM];
    cfc_inference_int8(sequence, sequence_output);

    cfc_reset_state();
    float step_output[CFC_OUTPUT_DIM];
    for (int t = 0; t < CFC_SEQ_LEN; t++) {
        cfc_single_step(sequence[t], step_output);
    }
    for (int i = 0; i < CFC_OUTPUT_DIM; i++) {
        if (!isfinite(step_output[i])) {
            fprintf(stderr, "non-finite stateful output at %d\n", i);
            return 2;
        }
        if (!close_enough(step_output[i], sequence_output[i], 1e-5f)) {
            fprintf(stderr, "persistent/sequence mismatch at %d\n", i);
            return 3;
        }
    }

    cfc_reset_state();
    float first_after_reset[CFC_OUTPUT_DIM];
    cfc_single_step(sequence[0], first_after_reset);
    cfc_reset_state();
    float repeated_first[CFC_OUTPUT_DIM];
    cfc_single_step(sequence[0], repeated_first);
    for (int i = 0; i < CFC_OUTPUT_DIM; i++) {
        if (!close_enough(first_after_reset[i], repeated_first[i], 1e-6f)) {
            fprintf(stderr, "reset is not deterministic at %d\n", i);
            return 4;
        }
    }

    puts("CfC PC golden test passed");
    return 0;
}
