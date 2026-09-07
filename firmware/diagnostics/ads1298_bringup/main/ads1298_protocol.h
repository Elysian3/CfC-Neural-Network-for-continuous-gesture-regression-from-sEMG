#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef ADS_FRAME_BYTES
#define ADS_FRAME_BYTES 27
#endif

#define ADS1298_CHANNEL_COUNT 8
#define ADS1298_STATUS_BYTES 3
#define ADS1298_CHANNEL_BYTES 3

// The supplied Windows upper computer uses these binary wire values. Its
// Buffer.BlockCopy call interprets waveform payloads as little-endian float32.
#define ADS1298_PC_FRAME_HEADER 0xA5U
#define ADS1298_PC_FRAME_TAIL 0x5AU
#define ADS1298_PC_COMMAND_HEADER 0xAAU
#define ADS1298_PC_COMMAND_TAIL 0xBBU
#define ADS1298_PC_FRAME_TYPE_SAMPLE_PARAMETERS 0x10U
#define ADS1298_PC_FRAME_TYPE_EMG 0x11U
#define ADS1298_PC_COMMAND_CONNECTION_STATUS 0x09U
#define ADS1298_PC_COMMAND_SAMPLE_PARAMETERS 0x10U
#define ADS1298_PC_COMMAND_START 0x11U
#define ADS1298_PC_MAX_SAMPLES_PER_PACKET 7U
#define ADS1298_PC_FLOAT_BYTES 4U
#define ADS1298_PC_SAMPLE_PARAMETERS_PACKET_BYTES 7U
#define ADS1298_PC_MAX_EMG_PACKET_BYTES 229U
#define ADS1298_PC_MAX_COMMAND_PACKET_BYTES 64U
#define ADS1298_PC_MAX_COMMAND_DATA_BYTES \
    (ADS1298_PC_MAX_COMMAND_PACKET_BYTES - 6U)

typedef struct {
    uint8_t status[ADS1298_STATUS_BYTES];
    int32_t channels[8];
} ads1298_frame_t;

typedef struct {
    uint8_t address;
    uint8_t data[ADS1298_PC_MAX_COMMAND_DATA_BYTES];
    size_t data_length;
} ads1298_pc_command_t;

typedef struct {
    uint8_t bytes[ADS1298_PC_MAX_COMMAND_PACKET_BYTES];
    size_t length;
} ads1298_pc_command_parser_t;

int32_t ads1298_decode_signed24(const uint8_t encoded[ADS1298_CHANNEL_BYTES]);
bool ads1298_decode_frame(const uint8_t raw[ADS_FRAME_BYTES], ads1298_frame_t *decoded);
float ads1298_code_to_microvolts(int32_t code, uint8_t gain);
size_t ads1298_pack_emg_frame(const ads1298_frame_t *samples, size_t sample_count,
                              uint8_t pga_gain, uint8_t *packet, size_t packet_capacity);
size_t ads1298_pack_sample_parameters(uint8_t rate_index, uint8_t pga_index,
                                       uint8_t *packet, size_t packet_capacity);
void ads1298_pc_command_parser_init(ads1298_pc_command_parser_t *parser);
bool ads1298_pc_command_parser_consume(ads1298_pc_command_parser_t *parser,
                                       uint8_t byte, ads1298_pc_command_t *command);
bool ads1298_protocol_self_test(void);
