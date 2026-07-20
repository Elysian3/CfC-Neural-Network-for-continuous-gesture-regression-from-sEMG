// features.h — RMS feature extraction with ring buffer
// 200 ms window, 50 ms stride, 2000 Hz sample rate
#pragma once
#include <stdint.h>
#include <math.h>

#define RMS_WINDOW_MS    200
#define RMS_STRIDE_MS    50
#define RMS_FS           2000
#define RMS_WINDOW_SAMPS ((RMS_WINDOW_MS * RMS_FS) / 1000)  // 400
#define RMS_STRIDE_SAMPS ((RMS_STRIDE_MS * RMS_FS) / 1000)  // 100
#define N_CHANNELS       12
#define N_FEATURES       12  // RMS-only = N_CHANNELS

// ── Ring buffer (one per channel) ──────────────────────────────────────────
typedef struct {
    float *buf;        // circular buffer, size = window_samps
    int    head;       // write position (next sample to overwrite)
    int    count;      // number of samples currently stored (≤ window_samps)
    float  sumsq;      // running sum of squares
} ringbuf_t;

// Initialise a ring buffer.  Caller owns the `buf` memory (window_samps floats).
void ringbuf_init(ringbuf_t *rb, float *buf);

// Push a sample, pop the oldest.  Returns current RMS value.
float ringbuf_update(ringbuf_t *rb, float sample);

// ── RMS feature extraction ─────────────────────────────────────────────────

// Extract RMS features from filtered EMG data.
//   filtered:  (n_samples, n_channels) row-major float array
//   features:  (max_windows, n_channels) output array
// Returns number of windows actually produced.
int extract_rms_features(const float *filtered,
                         int n_samples,
                         int n_channels,
                         float *features,
                         int max_windows);
