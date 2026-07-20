// features.c — RMS feature extraction with ring buffer
#include "features.h"
#include <string.h>

void ringbuf_init(ringbuf_t *rb, float *buf) {
    rb->buf   = buf;
    rb->head  = 0;
    rb->count = 0;
    rb->sumsq = 0.0f;
    memset(buf, 0, RMS_WINDOW_SAMPS * sizeof(float));
}

float ringbuf_update(ringbuf_t *rb, float sample) {
    // Pop oldest if buffer is full
    if (rb->count == RMS_WINDOW_SAMPS) {
        float old = rb->buf[rb->head];
        rb->sumsq -= old * old;
    } else {
        rb->count++;
    }

    // Push new sample
    rb->buf[rb->head] = sample;
    rb->sumsq += sample * sample;
    rb->head = (rb->head + 1) % RMS_WINDOW_SAMPS;

    // RMS = sqrt(sumsq / count)
    if (rb->count == 0) return 0.0f;
    return sqrtf(rb->sumsq / (float)rb->count);
}

int extract_rms_features(const float *filtered,
                         int n_samples,
                         int n_channels,
                         float *features,
                         int max_windows)
{
    // Per-channel ring buffers
    ringbuf_t rbs[N_CHANNELS];
    float buf[N_CHANNELS][RMS_WINDOW_SAMPS];
    for (int ch = 0; ch < n_channels; ch++)
        ringbuf_init(&rbs[ch], buf[ch]);

    int stride = RMS_STRIDE_SAMPS;
    int window_idx = 0;

    for (int s = 0; s < n_samples; s++) {
        // Update each channel's ring buffer
        for (int ch = 0; ch < n_channels; ch++) {
            ringbuf_update(&rbs[ch], filtered[s * n_channels + ch]);
        }

        // Extract feature at each stride boundary, once the first window is full
        if (s >= RMS_WINDOW_SAMPS - 1 && (s - (RMS_WINDOW_SAMPS - 1)) % stride == 0) {
            if (window_idx >= max_windows) break;
            for (int ch = 0; ch < n_channels; ch++) {
                features[window_idx * n_channels + ch] =
                    sqrtf(rbs[ch].sumsq / (float)RMS_WINDOW_SAMPS);
            }
            window_idx++;
        }
    }

    return window_idx;
}
