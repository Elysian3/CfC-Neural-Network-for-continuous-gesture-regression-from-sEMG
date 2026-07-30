// main.c — Antikythera ESP32-S3 DenseCfC RMS-only EMG-to-DoA Pipeline
// ============================================================================
// Hardware:  ESP32-S3 Zero + 74HC4051 12ch mux
// Pipeline:  ADC(12ch@2000Hz) → DSP(notch/hpf/lpf) → RMS(200ms/50ms)
//            → mu-law → CfC(h=256, 8-seq) → 5-DoA output
// SRAM:      Raw ringbuf 26.4KB + Filtered ringbuf 19.2KB + RMS seq 0.8KB
//            + CfC scratch ~5KB = ~52KB (weights remain in flash)
// ============================================================================

#include <stdint.h>
#include <string.h>
#include <math.h>

// ── ESP-IDF includes (available when building with idf.py) ─────────────────
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "driver/gpio.h"
#include "hal/adc_types.h"
#include "esp_adc/adc_continuous.h"
#include "esp_timer.h"
#include "dsps_biquad.h"
#include "dsps_biquad_gen.h"

// ── Project includes ───────────────────────────────────────────────────────
#include "weights.h"         // Per-channel INT8 CfC weights
#include "normalization.h"   // Mu-law center/scale/mu
#include "features.h"        // Ring buffer + RMS extraction
#include "cfc_inference.h"   // CfC single-step + sequence inference

// ── Constants ──────────────────────────────────────────────────────────────
#define ADC_FS          2000        // Sample rate per channel (Hz)
#define ADC_PERIOD_US   (1000000 / (ADC_FS * N_CHANNELS))  // ~41.7 µs per sample
#define MUX_SETTLE_US   10          // 74HC4051 settling time
#define DSP_PERIOD_MS   50          // Feature extraction period
#define RINGBUF_RAW_MS  550         // Raw ring buffer depth
#define RINGBUF_RAW_LEN ((RINGBUF_RAW_MS * ADC_FS) / 1000)  // 1100 samples/ch
#define RINGBUF_FLT_LEN RMS_WINDOW_SAMPS  // 400 samples/ch for 200ms window
#define ADC_READ_COUNT  (N_CHANNELS * 16)  // DMA conv_frame_size

// ── GPIO: 74HC4051 multiplexer ─────────────────────────────────────────────
#define MUX_SIGNAL_GPIO  GPIO_NUM_1
#define MUX_S0_GPIO      GPIO_NUM_17
#define MUX_S1_GPIO      GPIO_NUM_18
#define MUX_S2_GPIO      GPIO_NUM_8
#define MUX_MASK ((1ULL << MUX_S0_GPIO) | (1ULL << MUX_S1_GPIO) | (1ULL << MUX_S2_GPIO))

// ── SRAM Budget (all sizes in bytes) ────────────────────────────────────────
// raw_ringbuf:      RINGBUF_RAW_LEN × N_CHANNELS × 2 = 26,400 (int16_t)
// flt_ringbuf:      RINGBUF_FLT_LEN × N_CHANNELS × 4 = 19,200 (float)
// biquad_state:     12ch × 3 filters × (5 coeffs + 2 state) × 4 = 1,008
// rms_seq:          8 × N_FEATURES × 4 = 384
// model_weights:    6 layers INT8 weights = 163 KB in flash
// TOTAL INTERNAL:   ~52 KB of 400 KB SRAM budget

// ── Ring buffers ────────────────────────────────────────────────────────────
static int16_t raw_ringbuf[RINGBUF_RAW_LEN][N_CHANNELS];
static volatile int    raw_head = 0;     // ISR writes here

static float   flt_ringbuf[RINGBUF_FLT_LEN][N_CHANNELS];
static volatile int    flt_head = 0;     // DSP task writes here
static volatile int    flt_count = 0;

// ── DSP filter state (one biquad per channel per filter stage) ──────────────
static float notch_coeffs[12][5];  // b0,b1,b2,a1,a2
static float notch_w[12][2];       // delay line per channel
static float hpf_coeffs[12][5];
static float hpf_w[12][2];
static float lpf_coeffs[12][5];
static float lpf_w[12][2];

// ── RMS feature sequence (most recent 8 windows) ───────────────────────────
static float rms_seq[CFC_SEQ_LEN][N_FEATURES];
static int   rms_seq_idx = 0;

// ── Forward declarations ────────────────────────────────────────────────────
static TaskHandle_t dsp_task_handle = NULL;
static TaskHandle_t cfc_task_handle = NULL;
static void adc_isr_task(void *arg);
static void dsp_task(void *arg);
static void cfc_task(void *arg);
static void output_task(void *arg);
static void init_dsp_filters(void);
static void init_adc(void);
static void init_mux_gpio(void);

// ============================================================================
//  74HC4051 Multiplexer
// ============================================================================

static inline void mux_select(int channel) {
    // Drive S0-S2 for 74HC4051 channel selection (Y0–Y11)
    gpio_set_level(MUX_S0_GPIO, (channel >> 0) & 1);
    gpio_set_level(MUX_S1_GPIO, (channel >> 1) & 1);
    gpio_set_level(MUX_S2_GPIO, (channel >> 2) & 1);
    esp_rom_delay_us(MUX_SETTLE_US);  // Wait for signal to settle
}

static void init_mux_gpio(void) {
    gpio_config_t cfg = {
        .pin_bit_mask = MUX_MASK,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&cfg);
}

// ============================================================================
//  DSP Biquad Filter Chain: notch(50Hz) → HPF(20Hz) → LPF(450Hz)
// ============================================================================

static void init_dsp_filters(void) {
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        // All channels use identical filter coefficients
        dsps_biquad_gen_notch_f32(notch_coeffs[ch], 50.0f / (float)ADC_FS, 0.0f, 0.707f);
        dsps_biquad_gen_hpf_f32  (hpf_coeffs[ch],   20.0f / (float)ADC_FS, 0.707f);
        dsps_biquad_gen_lpf_f32  (lpf_coeffs[ch],   450.0f / (float)ADC_FS, 0.707f);

        memset(notch_w[ch], 0, sizeof(notch_w[ch]));
        memset(hpf_w[ch],   0, sizeof(hpf_w[ch]));
        memset(lpf_w[ch],   0, sizeof(lpf_w[ch]));
    }
}

// Process one sample per channel through the full filter chain
static float filter_sample(int ch, float raw) {
    float s = raw;
    dsps_biquad_f32(&s, &s, 1, notch_coeffs[ch], notch_w[ch]);
    dsps_biquad_f32(&s, &s, 1, hpf_coeffs[ch],   hpf_w[ch]);
    dsps_biquad_f32(&s, &s, 1, lpf_coeffs[ch],   lpf_w[ch]);
    return s;
}

// ============================================================================
//  ADC Continuous Mode (DMA-driven, 12 channels via 74HC4051)
// ============================================================================

static adc_continuous_handle_t adc_handle = NULL;

static void init_adc(void) {
    adc_continuous_handle_cfg_t adc_cfg = {
        .max_store_buf_size = 4096,
        .conv_frame_size    = ADC_READ_COUNT,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_cfg, &adc_handle));

    adc_digi_pattern_config_t adc_pat = {
        .atten    = ADC_ATTEN_DB_12,
        .channel  = ADC_CHANNEL_0,  // MUX signal input
        .unit     = ADC_UNIT_1,
        .bit_width = ADC_BITWIDTH_12,
    };

    adc_continuous_config_t dig_cfg = {
        .sample_freq_hz = ADC_FS * N_CHANNELS,  // 24,000 Hz total
        .conv_mode      = ADC_CONV_SINGLE_UNIT_1,
        .pattern_num    = 1,
        .adc_pattern    = &adc_pat,
    };
    ESP_ERROR_CHECK(adc_continuous_config(adc_handle, &dig_cfg));
    ESP_ERROR_CHECK(adc_continuous_start(adc_handle));
}

// ============================================================================
//  Task 1: ADC ISR → raw ring buffer (Core 1, highest priority)
// ============================================================================

static void adc_isr_task(void *arg) {
    (void)arg;
    uint8_t result[ADC_READ_COUNT];
    uint32_t out_len;

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        mux_select(ch);
        esp_rom_delay_us(MUX_SETTLE_US);

        ESP_ERROR_CHECK(adc_continuous_read(adc_handle, result,
                                            ADC_READ_COUNT / N_CHANNELS,
                                            &out_len, portMAX_DELAY));

        // Convert raw ADC to int16_t and write to ring buffer
        // Each channel gets a quick burst of samples per mux position
        for (int i = 0; i < (int)(out_len / SOC_ADC_DIGI_RESULT_BYTES); i++) {
            adc_digi_output_data_t *p =
                (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
            int16_t val = (int16_t)(p->type2.data & 0xFFF);  // 12-bit (ESP32-S3 uses type2)

            raw_ringbuf[raw_head][ch] = val;
        }
    }

    raw_head = (raw_head + 1) % RINGBUF_RAW_LEN;

    // Minimal delay for stable 2000 Hz per-channel rate
    vTaskDelay(pdMS_TO_TICKS(1));
    // Re-submit this task continuously (or use timer)
    // In production: this would be a proper ADC continuous ISR callback
    vTaskDelete(NULL);  // placeholder — will use ISR callback in production
    // FIXME: Replace with infinite loop + timer/ISR-driven sampling
}

// ============================================================================
//  Task 2: DSP filtering + RMS extraction (Core 0, medium priority)
//  Triggered every 50ms by FreeRTOS timer
// ============================================================================

static TimerHandle_t dsp_timer;

static void dsp_timer_cb(TimerHandle_t timer) {
    // Signal DSP task to process the latest 200ms window
    xTaskNotifyGive(dsp_task_handle);
}

static void dsp_task(void *arg) {
    (void)arg;
    static ringbuf_t rbs[N_CHANNELS];
    static float     buf[N_CHANNELS][RMS_WINDOW_SAMPS];   // ~19KB — must not live on stack

    for (int ch = 0; ch < N_CHANNELS; ch++)
        ringbuf_init(&rbs[ch], buf[ch]);

    for (;;) {
        // Wait for 50ms timer notification
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // Process the last RMS_STRIDE_SAMPS (100) raw samples through DSP
        int start = (raw_head - RMS_STRIDE_SAMPS + RINGBUF_RAW_LEN)
                    % RINGBUF_RAW_LEN;
        for (int s = 0; s < RMS_STRIDE_SAMPS; s++) {
            int idx = (start + s) % RINGBUF_RAW_LEN;
            float filtered[N_CHANNELS];
            for (int ch = 0; ch < N_CHANNELS; ch++) {
                filtered[ch] = filter_sample(ch,
                    (float)raw_ringbuf[idx][ch]);
            }

            // Write filtered samples to flt_ringbuf for RMS extraction
            for (int ch = 0; ch < N_CHANNELS; ch++)
                flt_ringbuf[flt_head][ch] = filtered[ch];
            flt_head = (flt_head + 1) % RINGBUF_FLT_LEN;
            if (flt_count < RINGBUF_FLT_LEN) flt_count++;

            // Update RMS ring buffers
            for (int ch = 0; ch < N_CHANNELS; ch++)
                ringbuf_update(&rbs[ch], filtered[ch]);
        }

        // Extract RMS features from current ring buffer state
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            rms_seq[rms_seq_idx][ch] =
                sqrtf(rbs[ch].sumsq / (float)RMS_WINDOW_SAMPS);
        }

        // Apply mu-law normalization
        mulaw_encode(rms_seq[rms_seq_idx], N_FEATURES);

        // Advance sequence index (circular 8-frame buffer)
        rms_seq_idx = (rms_seq_idx + 1) % CFC_SEQ_LEN;

        // Notify CfC task
        xTaskNotifyGive(cfc_task_handle);
    }
}

// ============================================================================
//  Task 3: CfC inference (Core 0, low priority)
//  Runs after each new RMS frame is available
// ============================================================================

static void cfc_task(void *arg) {
    (void)arg;

    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // Latest RMS frame (one before current write head)
        int idx = (rms_seq_idx - 1 + CFC_SEQ_LEN) % CFC_SEQ_LEN;

        // Single-step CfC RNN inference — persistent hidden state
        float output[CFC_OUTPUT_DIM];
        uint32_t t_start = esp_timer_get_time();
        cfc_single_step(rms_seq[idx], output);
        uint32_t t_elapsed = esp_timer_get_time() - t_start;

        // Log latency
        printf("LATENCY: %lu us | DoA: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
               t_elapsed,
               output[0], output[1], output[2], output[3], output[4]);
    }
}

// ============================================================================
//  Task 4: Output (Core 0, lowest priority)
// ============================================================================

static void output_task(void *arg) {
    (void)arg;

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(DSP_PERIOD_MS));
        // Future: I2C to motor driver, PWM, or BLE HID output
        // For now: CfC task handles serial output directly
    }
}

// ============================================================================
//  Main: initialise and start FreeRTOS scheduler
// ============================================================================

void app_main(void) {
    printf("\n=== Antikythera ESP32-S3 DenseCfC RMS-only ===\n");
    printf("Weight backend: Flash direct, persistent single-step\n");
    printf("Internal SRAM budget: ~52 KB / 400 KB\n");
    printf("Pipeline: ADC(12ch@2kHz) → DSP → RMS → mu-law → CfC → 5-DoA\n");
    printf("==============================================\n\n");

    // 1. Init hardware
    init_mux_gpio();
    init_adc();
    init_dsp_filters();
    cfc_lut_init();

    // 2. Create 50ms DSP timer
    dsp_timer = xTimerCreate(
        "dsp_timer",
        pdMS_TO_TICKS(DSP_PERIOD_MS),
        pdTRUE,   // auto-reload
        NULL,
        dsp_timer_cb
    );

    // 3. Create FreeRTOS tasks
    //    Task          | Core | Priority | Stack  | Handle
    //    --------------|------|----------|--------|--------
    //    ADC ISR task  |  1   |  highest | 4096   | (timer-driven)
    //    DSP task      |  0   |  medium  | 8192   | dsp_task_handle
    //    CfC task      |  0   |  low     | 12288  | cfc_task_handle
    //    Output task   |  0   |  lowest  | 2048   | (none)

    xTaskCreatePinnedToCore(dsp_task,    "dsp",    8192, NULL, 5,
                            &dsp_task_handle, 0);
    xTaskCreatePinnedToCore(cfc_task,    "cfc",    12288, NULL, 3,
                            &cfc_task_handle, 0);
    xTaskCreatePinnedToCore(output_task, "output", 2048, NULL, 1,
                            NULL, 0);

    // 4. Start the ADC task on Core 1 (highest priority)
    //    In production: replace with proper ADC continuous ISR
    xTaskCreatePinnedToCore(adc_isr_task, "adc", 4096, NULL, 7,
                            NULL, 1);

    // 5. Start DSP timer
    xTimerStart(dsp_timer, 0);

    printf("FreeRTOS scheduler running...\n");
    // Scheduler takes over from here
}
