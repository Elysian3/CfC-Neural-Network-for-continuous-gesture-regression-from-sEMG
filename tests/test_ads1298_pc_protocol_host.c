#include "ads1298_protocol.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static float read_float32_le(const uint8_t *encoded)
{
    uint32_t bits = (uint32_t)encoded[0] |
                    ((uint32_t)encoded[1] << 8) |
                    ((uint32_t)encoded[2] << 16) |
                    ((uint32_t)encoded[3] << 24);
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void test_emg_frame_contains_little_endian_microvolts(void)
{
    ads1298_frame_t sample = {0};
    uint8_t packet[ADS1298_PC_MAX_EMG_PACKET_BYTES] = {0};

    sample.channels[0] = 0;
    sample.channels[1] = 0x7FFFFF;
    sample.channels[2] = -0x800000;

    const size_t length = ads1298_pack_emg_frame(&sample, 1, 6U, packet, sizeof(packet));
    assert(length == 37U);
    assert(packet[0] == 0xA5U);
    assert(packet[1] == 34U);
    assert(packet[2] == ADS1298_PC_FRAME_TYPE_EMG);
    assert(packet[3] == (uint8_t)(packet[1] ^ packet[2]));
    assert(packet[length - 1U] == 0x5AU);
    assert(fabsf(read_float32_le(&packet[4])) < 0.001f);
    assert(fabsf(read_float32_le(&packet[8]) - 400000.0f) < 1.0f);
    assert(fabsf(read_float32_le(&packet[12]) + 400000.0f) < 1.0f);
}

static void test_emg_packet_batches_no_more_than_seven_samples(void)
{
    ads1298_frame_t samples[ADS1298_PC_MAX_SAMPLES_PER_PACKET] = {0};
    uint8_t packet[ADS1298_PC_MAX_EMG_PACKET_BYTES] = {0};

    assert(ads1298_pack_emg_frame(samples, ADS1298_PC_MAX_SAMPLES_PER_PACKET, 6U,
                                   packet, sizeof(packet)) == 229U);
    assert(packet[1] == 226U);
    assert(ads1298_pack_emg_frame(samples, ADS1298_PC_MAX_SAMPLES_PER_PACKET + 1U, 6U,
                                   packet, sizeof(packet)) == 0U);
}

static void test_device_commands_survive_fragmented_serial_input(void)
{
    static const uint8_t start_command[] = {0xAA, 0x04, 0x80, 0x11, 0x01, 0x94, 0xBB};
    static const uint8_t parameter_command[] = {
        0xAA, 0x06, 0x80, 0x10, 0x02, 0x04, 0x00, 0x90, 0xBB,
    };
    ads1298_pc_command_parser_t parser;
    ads1298_pc_command_t command;

    ads1298_pc_command_parser_init(&parser);
    for (size_t index = 0; index < sizeof(start_command) - 1U; index++) {
        assert(!ads1298_pc_command_parser_consume(&parser, start_command[index], &command));
    }
    assert(ads1298_pc_command_parser_consume(&parser, start_command[sizeof(start_command) - 1U],
                                             &command));
    assert(command.address == ADS1298_PC_COMMAND_START);
    assert(command.data_length == 1U);
    assert(command.data[0] == 1U);

    for (size_t index = 0; index < sizeof(parameter_command); index++) {
        const bool complete = ads1298_pc_command_parser_consume(
            &parser, parameter_command[index], &command);
        if (index + 1U == sizeof(parameter_command)) {
            assert(complete);
        } else {
            assert(!complete);
        }
    }
    assert(command.address == ADS1298_PC_COMMAND_SAMPLE_PARAMETERS);
    assert(command.data_length == 3U);
    assert(command.data[0] == 2U);
    assert(command.data[1] == 4U);
}

static void test_sample_parameter_reply_describes_actual_ads_configuration(void)
{
    uint8_t packet[ADS1298_PC_SAMPLE_PARAMETERS_PACKET_BYTES] = {0};
    const size_t length = ads1298_pack_sample_parameters(2U, 4U, packet, sizeof(packet));

    assert(length == ADS1298_PC_SAMPLE_PARAMETERS_PACKET_BYTES);
    assert(packet[0] == 0xA5U);
    assert(packet[1] == 4U);
    assert(packet[2] == ADS1298_PC_FRAME_TYPE_SAMPLE_PARAMETERS);
    assert(packet[3] == (uint8_t)(packet[1] ^ packet[2]));
    assert(packet[4] == 2U);
    assert(packet[5] == 4U);
    assert(packet[6] == 0x5AU);
}

int main(void)
{
    test_emg_frame_contains_little_endian_microvolts();
    test_emg_packet_batches_no_more_than_seven_samples();
    test_device_commands_survive_fragmented_serial_input();
    test_sample_parameter_reply_describes_actual_ads_configuration();
    return 0;
}
