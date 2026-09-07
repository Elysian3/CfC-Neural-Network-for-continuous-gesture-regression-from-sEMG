#include "ads1298_protocol.h"

#include <limits.h>
#include <string.h>

#define ADS_SIGNED24_MASK 0xFFFFFFU
#define ADS1298_INTERNAL_REFERENCE_UV 2400000.0f
#define ADS1298_SIGNED24_MAX_CODE 8388607.0f

_Static_assert(sizeof(float) == ADS1298_PC_FLOAT_BYTES,
               "PC waveform protocol requires IEEE-754 float32");

int32_t ads1298_decode_signed24(const uint8_t encoded[ADS1298_CHANNEL_BYTES])
{
    uint32_t value = ((uint32_t)encoded[0] << 16) |
                     ((uint32_t)encoded[1] << 8) |
                     (uint32_t)encoded[2];

    if ((value & 0x800000U) != 0U) {
        value |= 0xFF000000U;
    }
    return (int32_t)value;
}

bool ads1298_decode_frame(const uint8_t raw[ADS_FRAME_BYTES], ads1298_frame_t *decoded)
{
    if (raw == NULL || decoded == NULL) {
        return false;
    }

    memcpy(decoded->status, raw, ADS1298_STATUS_BYTES);
    // Output order is channels[8]: CH1, CH2, ... CH8.
    for (size_t channel = 0; channel < ADS1298_CHANNEL_COUNT; channel++) {
        size_t offset = ADS1298_STATUS_BYTES + (channel * ADS1298_CHANNEL_BYTES);
        decoded->channels[channel] = ads1298_decode_signed24(&raw[offset]);
    }
    return true;
}

float ads1298_code_to_microvolts(int32_t code, uint8_t gain)
{
    if (gain == 0U) {
        return 0.0f;
    }
    return ((float)code * ADS1298_INTERNAL_REFERENCE_UV) /
           ((float)gain * ADS1298_SIGNED24_MAX_CODE);
}

static void ads1298_store_float32_le(float value, uint8_t encoded[ADS1298_PC_FLOAT_BYTES])
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    encoded[0] = (uint8_t)(bits & 0xFFU);
    encoded[1] = (uint8_t)((bits >> 8) & 0xFFU);
    encoded[2] = (uint8_t)((bits >> 16) & 0xFFU);
    encoded[3] = (uint8_t)((bits >> 24) & 0xFFU);
}

size_t ads1298_pack_emg_frame(const ads1298_frame_t *samples, size_t sample_count,
                              uint8_t pga_gain, uint8_t *packet, size_t packet_capacity)
{
    if (samples == NULL || packet == NULL || pga_gain == 0U || sample_count == 0U ||
        sample_count > ADS1298_PC_MAX_SAMPLES_PER_PACKET) {
        return 0U;
    }

    const size_t float_count = sample_count * ADS1298_CHANNEL_COUNT;
    const size_t payload_bytes = float_count * ADS1298_PC_FLOAT_BYTES;
    const size_t frame_length_field = 2U + payload_bytes;
    const size_t packet_bytes = frame_length_field + 3U;
    if (packet_capacity < packet_bytes || frame_length_field > UINT8_MAX) {
        return 0U;
    }

    packet[0] = ADS1298_PC_FRAME_HEADER;
    packet[1] = (uint8_t)frame_length_field;
    packet[2] = ADS1298_PC_FRAME_TYPE_EMG;
    packet[3] = packet[1] ^ packet[2];

    size_t output_offset = 4U;
    for (size_t sample = 0; sample < sample_count; sample++) {
        for (size_t channel = 0; channel < ADS1298_CHANNEL_COUNT; channel++) {
            const float microvolts = ads1298_code_to_microvolts(samples[sample].channels[channel],
                                                                 pga_gain);
            ads1298_store_float32_le(microvolts, &packet[output_offset]);
            output_offset += ADS1298_PC_FLOAT_BYTES;
        }
    }

    packet[output_offset] = ADS1298_PC_FRAME_TAIL;
    return packet_bytes;
}

size_t ads1298_pack_sample_parameters(uint8_t rate_index, uint8_t pga_index,
                                       uint8_t *packet, size_t packet_capacity)
{
    if (packet == NULL || packet_capacity < ADS1298_PC_SAMPLE_PARAMETERS_PACKET_BYTES) {
        return 0U;
    }

    packet[0] = ADS1298_PC_FRAME_HEADER;
    packet[1] = 4U;
    packet[2] = ADS1298_PC_FRAME_TYPE_SAMPLE_PARAMETERS;
    packet[3] = packet[1] ^ packet[2];
    packet[4] = rate_index;
    packet[5] = pga_index;
    packet[6] = ADS1298_PC_FRAME_TAIL;
    return ADS1298_PC_SAMPLE_PARAMETERS_PACKET_BYTES;
}

void ads1298_pc_command_parser_init(ads1298_pc_command_parser_t *parser)
{
    if (parser != NULL) {
        parser->length = 0U;
    }
}

static void ads1298_pc_command_parser_reset(ads1298_pc_command_parser_t *parser)
{
    parser->length = 0U;
}

bool ads1298_pc_command_parser_consume(ads1298_pc_command_parser_t *parser,
                                       uint8_t byte, ads1298_pc_command_t *command)
{
    if (parser == NULL || command == NULL) {
        return false;
    }

    if (parser->length == 0U && byte != ADS1298_PC_COMMAND_HEADER) {
        return false;
    }
    if (parser->length >= sizeof(parser->bytes)) {
        ads1298_pc_command_parser_reset(parser);
        return false;
    }

    parser->bytes[parser->length++] = byte;
    if (parser->length == 2U) {
        const size_t total_bytes = (size_t)parser->bytes[1] + 3U;
        if (parser->bytes[1] < 3U || total_bytes > sizeof(parser->bytes)) {
            ads1298_pc_command_parser_reset(parser);
        }
        return false;
    }
    if (parser->length < 2U) {
        return false;
    }

    const size_t total_bytes = (size_t)parser->bytes[1] + 3U;
    if (parser->length < total_bytes) {
        return false;
    }
    if (parser->length != total_bytes) {
        ads1298_pc_command_parser_reset(parser);
        return false;
    }

    uint8_t checksum = 0U;
    for (size_t index = 1U; index < total_bytes - 2U; index++) {
        checksum ^= parser->bytes[index];
    }
    const size_t data_length = (size_t)parser->bytes[1] - 3U;
    const bool valid = parser->bytes[0] == ADS1298_PC_COMMAND_HEADER &&
                       parser->bytes[2] == 0x80U &&
                       parser->bytes[total_bytes - 2U] == checksum &&
                       parser->bytes[total_bytes - 1U] == ADS1298_PC_COMMAND_TAIL &&
                       data_length <= ADS1298_PC_MAX_COMMAND_DATA_BYTES;
    if (valid) {
        command->address = parser->bytes[3];
        command->data_length = data_length;
        if (data_length > 0U) {
            memcpy(command->data, &parser->bytes[4], data_length);
        }
    }
    ads1298_pc_command_parser_reset(parser);
    return valid;
}

bool ads1298_protocol_self_test(void)
{
    static const uint8_t zero[ADS1298_CHANNEL_BYTES] = {0x00, 0x00, 0x00};
    static const uint8_t largest_positive[ADS1298_CHANNEL_BYTES] = {0x7F, 0xFF, 0xFF};
    static const uint8_t smallest_negative[ADS1298_CHANNEL_BYTES] = {0x80, 0x00, 0x00};
    static const uint8_t minus_one[ADS1298_CHANNEL_BYTES] = {0xFF, 0xFF, 0xFF};

    return ads1298_decode_signed24(zero) == 0x000000 &&
           ads1298_decode_signed24(largest_positive) == 0x7FFFFF &&
           ads1298_decode_signed24(smallest_negative) == (INT32_MIN / 256) &&
           ((uint32_t)ads1298_decode_signed24(minus_one) & ADS_SIGNED24_MASK) ==
               ADS_SIGNED24_MASK &&
           ads1298_decode_signed24(minus_one) == -1;
}
