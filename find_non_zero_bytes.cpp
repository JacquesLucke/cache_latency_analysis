#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <cstring>
#include <chrono>
#include <sstream>
#include <numeric>
#include <emmintrin.h>
#include <immintrin.h>
#include <nmmintrin.h>
#include <assert.h>
#include <fstream>

static void find_indices__naive(uint8_t *__restrict in_begin,
                                uint8_t *__restrict in_end,
                                uint32_t *__restrict out_begin,
                                uint32_t &out_amount)
{
    uint32_t *out_current = out_begin;

    for (uint8_t *current = in_begin; current != in_end; current++) {
        if (*current != 0) {
            uint32_t index = current - in_begin;
            *out_current++ = index;
        }
    }

    out_amount = out_current - out_begin;
}

static void find_indices__branch_free(uint8_t *__restrict in_begin,
                                      uint8_t *__restrict in_end,
                                      uint32_t *__restrict out_begin,
                                      uint32_t &out_amount)
{
    uint32_t *out_current = out_begin;

    uint32_t amount = in_end - in_begin;
    for (uint32_t index = 0; index < amount; index++) {
        *out_current = index;
        uint8_t value = in_begin[index];
        bool is_non_zero = value != 0;
        out_current += is_non_zero;
    }

    out_amount = out_current - out_begin;
}

static void find_indices__grouped_2(uint8_t *__restrict in_begin,
                                    uint8_t *__restrict in_end,
                                    uint32_t *__restrict out_begin,
                                    uint32_t &out_amount)
{
    assert((in_end - in_begin) % 2 == 0);

    uint32_t *out_current = out_begin;

    for (uint8_t *current = in_begin; current != in_end; current += 2) {
        uint16_t group = *(uint16_t *)current;
        if (group != 0) {
            uint32_t index = current - in_begin;
            if (current[0]) {
                *out_current++ = index;
            }
            if (current[1]) {
                *out_current++ = index + 1;
            }
        }
    }

    out_amount = out_current - out_begin;
}

static void find_indices__grouped_4(uint8_t *__restrict in_begin,
                                    uint8_t *__restrict in_end,
                                    uint32_t *__restrict out_begin,
                                    uint32_t &out_amount)
{
    assert((in_end - in_begin) % 4 == 0);

    uint32_t *out_current = out_begin;

    for (uint8_t *current = in_begin; current != in_end; current += 4) {
        uint32_t group = *(uint32_t *)current;
        if (group != 0) {
            for (uint32_t i = 0; i < 4; i++) {
                if (current[i] != 0) {
                    *out_current++ = (current - in_begin) + i;
                }
            }
        }
    }

    out_amount = out_current - out_begin;
}

static void find_indices__grouped_8(uint8_t *__restrict in_begin,
                                    uint8_t *__restrict in_end,
                                    uint32_t *__restrict out_begin,
                                    uint32_t &out_amount)
{
    assert((in_end - in_begin) % 8 == 0);

    uint32_t *out_current = out_begin;

    for (uint8_t *current = in_begin; current != in_end; current += 8) {
        uint64_t group = *(uint64_t *)current;
        if (group != 0) {
            for (uint32_t i = 0; i < 8; i++) {
                if (current[i] != 0) {
                    *out_current++ = (current - in_begin) + i;
                }
            }
        }
    }

    out_amount = out_current - out_begin;
}

static uint32_t find_lowest_set_bit_index(uint32_t x)
{
    unsigned long index;
    assert(_BitScanForward(&index, x));
    return index;
}

static uint32_t find_highest_set_bit_index(uint32_t x)
{
    unsigned long index;
    assert(_BitScanReverse(&index, x));
    return index;
}

static uint32_t find_lowest_set_bit_index(uint64_t x)
{
    unsigned long index;
    assert(_BitScanForward64(&index, x));
    return index;
}

static uint32_t find_highest_set_bit_index(uint64_t x)
{
    unsigned long index;
    assert(_BitScanReverse64(&index, x));
    return index;
}

static void find_indices__optimized(uint8_t *__restrict in_begin,
                                    uint8_t *__restrict in_end,
                                    uint32_t *__restrict out_begin,
                                    uint32_t &out_amount)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros;
    __m256i ones;
    memset(&zeros, 0x00, sizeof(__m256i));
    memset(&ones, 0xFF, sizeof(__m256i));

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i compared = _mm256_cmpeq_epi8(group, zeros);
        compared = _mm256_andnot_si256(compared, ones);
        uint32_t mask = _mm256_movemask_epi8(compared);
        if (mask != 0) {
            uint32_t set_bits = _mm_popcnt_u32(mask);
            uint32_t index_offset = current - in_begin;
            switch (set_bits) {
                case 1: {
                    uint32_t index = find_lowest_set_bit_index(mask);
                    *out_current = index_offset + index;
                    out_current++;
                    break;
                }
                case 2: {
                    uint32_t index1 = find_lowest_set_bit_index(mask);
                    uint32_t index2 = find_highest_set_bit_index(mask);
                    *out_current = index_offset + index1;
                    *(out_current + 1) = index_offset + index2;
                    out_current += 2;
                    break;
                }
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                case 16:
                case 17:
                case 18:
                case 19:
                case 20:
                case 21:
                case 22:
                case 23:
                case 24:
                case 25:
                case 26:
                case 27:
                case 28:
                case 29:
                case 30:
                case 31: {
                    uint32_t index_end = index_offset + 32;
                    for (uint32_t index = index_offset; index < index_end;
                         index++) {
                        *out_current = index;
                        bool is_non_zero = in_begin[index] != 0;
                        out_current += is_non_zero;
                    }
                    break;
                }
                case 32: {
                    __m256i index_offset_256 = _mm256_set1_epi32(index_offset);
                    __m256i part1 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                    __m256i part2 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8));
                    __m256i part3 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(23, 22, 21, 20, 19, 18, 17, 16));
                    __m256i part4 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(31, 30, 29, 28, 27, 26, 25, 24));
                    _mm256_store_si256((__m256i *)out_current, part1);
                    _mm256_store_si256((__m256i *)out_current + 1, part2);
                    _mm256_store_si256((__m256i *)out_current + 2, part3);
                    _mm256_store_si256((__m256i *)out_current + 3, part4);
                    out_current += 32;
                    break;
                }
            }
        }
    }

    out_amount = out_current - out_begin;
}

static void find_indices__optimized2(uint8_t *__restrict in_begin,
                                     uint8_t *__restrict in_end,
                                     uint32_t *__restrict out_begin,
                                     uint32_t &out_amount)
{
    assert((in_end - in_begin) % 64 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros;
    memset(&zeros, 0x00, sizeof(__m256i));

    for (uint8_t *current = in_begin; current != in_end; current += 64) {
        __m256i group1 = _mm256_loadu_si256((__m256i *)current);
        __m256i group2 = _mm256_loadu_si256((__m256i *)current + 1);
        __m256i compared1 = _mm256_cmpeq_epi8(group1, zeros);
        __m256i compared2 = _mm256_cmpeq_epi8(group2, zeros);
        uint32_t inverted_mask1 = _mm256_movemask_epi8(compared1);
        uint32_t inverted_mask2 = _mm256_movemask_epi8(compared2);
        uint64_t inverted_mask = ((uint64_t)inverted_mask2 << (uint64_t)32) |
                                 inverted_mask1;
        uint64_t mask = ~inverted_mask;
        if (mask != 0) {
            uint32_t set_bits = _mm_popcnt_u64(mask);
            uint32_t start_index = current - in_begin;

            switch (set_bits) {
                case 1: {
                    uint32_t index = find_lowest_set_bit_index(mask);
                    *out_current = start_index + index;
                    out_current++;
                    break;
                }
                case 2: {
                    uint32_t index1 = find_lowest_set_bit_index(mask);
                    uint32_t index2 = find_highest_set_bit_index(mask);
                    *out_current = start_index + index1;
                    *(out_current + 1) = start_index + index2;
                    out_current += 2;
                    break;
                }
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                case 16:
                case 17:
                case 18:
                case 19:
                case 20:
                case 21:
                case 22:
                case 23:
                case 24:
                case 25:
                case 26:
                case 27:
                case 28:
                case 29:
                case 30:
                case 31:
                case 32: {
                    unsigned long index_offset;
                    while (_BitScanForward64(&index_offset, mask)) {
                        *out_current = start_index + index_offset;
                        out_current++;
                        mask &= ~((uint64_t)1 << index_offset);
                    }
                    break;
                }
                case 33:
                case 34:
                case 35:
                case 36:
                case 37:
                case 38:
                case 39:
                case 40:
                case 41:
                case 42:
                case 43:
                case 44:
                case 45:
                case 46:
                case 47:
                case 48:
                case 49:
                case 50:
                case 51:
                case 52:
                case 53:
                case 54:
                case 55:
                case 56:
                case 57:
                case 58:
                case 59:
                case 60:
                case 61:
                case 62:
                case 63: {
                    uint32_t index = start_index;
                    while (mask) {
                        *out_current = index;
                        out_current += mask & 1;
                        mask >>= 1;
                    }
                    break;
                }
                case 64: {
                    __m256i index_offset_256 = _mm256_set1_epi32(start_index);

                    __m256i part1 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
                    __m256i part2 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8));
                    __m256i part3 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(23, 22, 21, 20, 19, 18, 17, 16));
                    __m256i part4 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(31, 30, 29, 28, 27, 26, 25, 24));

                    __m256i part5 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(39, 38, 37, 36, 35, 34, 33, 32));
                    __m256i part6 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(47, 46, 45, 44, 43, 42, 41, 40));
                    __m256i part7 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(55, 54, 53, 52, 51, 50, 49, 48));
                    __m256i part8 = _mm256_add_epi32(
                        index_offset_256,
                        _mm256_set_epi32(63, 62, 61, 60, 59, 58, 57, 56));

                    _mm256_store_si256((__m256i *)out_current, part1);
                    _mm256_store_si256((__m256i *)out_current + 1, part2);
                    _mm256_store_si256((__m256i *)out_current + 2, part3);
                    _mm256_store_si256((__m256i *)out_current + 3, part4);
                    _mm256_store_si256((__m256i *)out_current + 5, part5);
                    _mm256_store_si256((__m256i *)out_current + 6, part6);
                    _mm256_store_si256((__m256i *)out_current + 7, part7);
                    _mm256_store_si256((__m256i *)out_current + 8, part8);
                    out_current += 64;
                    break;
                }
            }
        }
    }

    out_amount = out_current - out_begin;
}

static void find_indices__grouped_32(uint8_t *__restrict in_begin,
                                     uint8_t *__restrict in_end,
                                     uint32_t *__restrict out_begin,
                                     uint32_t &out_amount)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros;
    __m256i ones;
    memset(&zeros, 0x00, sizeof(__m256i));
    memset(&ones, 0xFF, sizeof(__m256i));

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i compared = _mm256_cmpeq_epi8(group, zeros);
        compared = _mm256_andnot_si256(compared, ones);
        uint32_t mask = _mm256_movemask_epi8(compared);
        if (mask != 0) {
            for (uint32_t i = 0; i < 32; i++) {
                if (current[i] != 0) {
                    *out_current++ = (current - in_begin) + i;
                }
            }
        }
    }

    out_amount = out_current - out_begin;
}

typedef void (*NonZeroFinder)(uint8_t *__restrict in_begin,
                              uint8_t *__restrict in_end,
                              uint32_t *__restrict out_begin,
                              uint32_t &out_amount);

static void print_found(std::vector<uint32_t> &found, uint32_t amount_found)
{
    std::cout << "Found:\n  ";
    for (uint32_t i = 0; i < amount_found; i++) {
        uint32_t index = found[i];
        std::cout << index << " ";
    }
    std::cout << '\n';
}

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Nanoseconds = std::chrono::nanoseconds;

struct FunctionStats {
    double min_ms;
    double max_ms;
    double average_ms;
};

static FunctionStats run_test(const char *name,
                              NonZeroFinder function,
                              std::vector<uint8_t> &array)
{
    const uint32_t iteration_count = 300;
    const uint32_t cutoff = 20;

    std::vector<uint32_t> found(array.size());
    uint32_t amount_found = 0;
    std::vector<double> durations;

    for (uint32_t iteration = 0; iteration < 100; iteration++) {
        TimePoint start_time = Clock::now();
        function(array.data(),
                 array.data() + array.size(),
                 &found[0],
                 amount_found);
        TimePoint end_time = Clock::now();
        Nanoseconds duration = end_time - start_time;
        durations.push_back(duration.count() / 1'000'000.0);
    }

    std::sort(durations.begin(), durations.end());

    double min_duration = *std::min_element(durations.begin() + cutoff,
                                            durations.end() - cutoff);
    double max_duration = *std::max_element(durations.begin() + cutoff,
                                            durations.end() - cutoff);
    double average_duration = std::accumulate(durations.begin() + cutoff,
                                              durations.end() - cutoff,
                                              0.0) /
                              (double)(durations.size() - cutoff * 2);
    std::cout << "Min Duration: " << min_duration << " ms (" << name << ")\n";
    std::cout << "Avg Duration: " << average_duration << " ms (" << name
              << ")\n";
    std::cout << "Max Duration: " << max_duration << " ms (" << name << ")\n";

    // print_found(found, amount_found);
    std::cout << "Found " << amount_found << " non-zeros.\n";
    return {min_duration, max_duration, average_duration};
}

static std::vector<uint8_t> init_data(uint32_t total_size, uint32_t one_amount)
{
    assert(one_amount <= total_size);

    std::vector<uint8_t> data(total_size);

    std::random_device rd;
    std::uniform_real_distribution<double> distribution(0, 1);
    uint32_t remaining = one_amount;

    for (uint32_t i = 0; i < total_size; i++) {
        double probability = (double)remaining / (double)(total_size - i);
        double random_value = distribution(rd);
        if (random_value < probability) {
            data[i] = 1;
            remaining--;
        }
        else {
            data[i] = 0;
        }
    }

    return data;
}

struct TestFunction {
    NonZeroFinder function;
    const char *name;
};

int main(int argc, char const *argv[])
{
    uint32_t total_amount = 10'000'000;
    std::cout << "Total size: " << total_amount << "\n";

    std::stringstream ss;

    std::vector<uint32_t> set_amounts = {0,
                                         1'000,
                                         100'000,
                                         1'000'000,
                                         5'000'000,
                                         9'000'000,
                                         9'900'000,
                                         9'990'000,
                                         10'000'000};

    std::vector<TestFunction> test_functions = {
        {find_indices__naive, "naive"},
        {find_indices__branch_free, "branch free"},
        {find_indices__grouped_2, "grouped 2"},
        {find_indices__grouped_4, "grouped 4"},
        {find_indices__grouped_8, "grouped 8"},
        {find_indices__grouped_32, "grouped 32"},
        {find_indices__optimized, "optimized"},
        {find_indices__optimized2, "optimized 2"},
    };

    for (uint32_t set_amount : set_amounts) {
        auto array = init_data(total_amount, set_amount);
        std::cout << "Set Amount: " << set_amount << "\n\n";

        for (auto test_function : test_functions) {
            ss << set_amount << ';';
            ss << test_function.name << ';';

            auto stats = run_test(
                test_function.name, test_function.function, array);

            ss << stats.min_ms << ';' << stats.max_ms << ';'
               << stats.average_ms << '\n';
        }

        std::cout << "\n\n";
    }

    std::ofstream file("C:\\Users\\jacques\\Documents\\benchmark_results.csv");
    file << ss.str();
    file.close();

    return 0;
}
