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
#include <algorithm>

static uint32_t find_lowest_set_bit_index(uint x)
{
    return (uint32_t)__builtin_ctz(x);
}

static uint32_t find_highest_set_bit_index(uint32_t x)
{
    return (uint32_t)__builtin_clz(x);
}

static uint32_t find_non_zero_indices__baseline(uint8_t *__restrict in_begin,
                                                uint8_t *in_end,
                                                uint32_t *__restrict out_begin)
{
    uint32_t *out_current = out_begin;

    for (uint8_t *current = in_begin; current != in_end; current++) {
        if (*current != 0) {
            uint32_t index = current - in_begin;
            *out_current = index;
            out_current++;
        }
    }

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__branch_free(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
{
    uint32_t *out_current = out_begin;

    uint32_t amount = in_end - in_begin;

    for (uint32_t index = 0; index < amount; index++) {
        *out_current = index;
        bool is_non_zero = in_begin[index] != 0;
        out_current += is_non_zero;
    }

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__grouped_branch_free(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros = _mm256_set1_epi8(0);

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i is_zero_byte_mask = _mm256_cmpeq_epi8(group, zeros);
        uint32_t is_zero_mask = _mm256_movemask_epi8(is_zero_byte_mask);
        uint32_t is_non_zero_mask = ~is_zero_mask;
        if (is_non_zero_mask != 0) {
            uint32_t start_index = current - in_begin;
            uint32_t end_index = start_index + 32;
            for (uint32_t index = start_index; index < end_index; index++) {
                *out_current = index;
                bool is_non_zero = in_begin[index] != 0;
                out_current += is_non_zero;
            }
        }
    }

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__grouped_branch_free_2(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros = _mm256_set1_epi8(0);

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i is_zero_byte_mask = _mm256_cmpeq_epi8(group, zeros);
        uint32_t is_zero_mask = _mm256_movemask_epi8(is_zero_byte_mask);
        uint32_t is_non_zero_mask = ~is_zero_mask;
        if (is_non_zero_mask != 0) {
            uint32_t index = current - in_begin;
            for (uint32_t i = 0; i < 32; i++) {
                *out_current = index;
                out_current += is_non_zero_mask & 1;
                index++;
                is_non_zero_mask >>= 1;
            }
        }
    }

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__mostly_ones(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros = _mm256_set1_epi8(0);

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i is_zero_byte_mask = _mm256_cmpeq_epi8(group, zeros);
        uint32_t is_zero_mask = _mm256_movemask_epi8(is_zero_byte_mask);
        if (is_zero_mask != 0xFFFFFFFF) {
            if (is_zero_mask != 0x00000000) {
                /* Group has zeros and ones, use branchless algorithm. */
                uint32_t is_non_zero_mask = ~is_zero_mask;
                uint32_t index = current - in_begin;
                for (uint32_t i = 0; i < 32; i++) {
                    *out_current = index;
                    out_current += is_non_zero_mask & 1;
                    index++;
                    is_non_zero_mask >>= 1;
                }
            }
            else {
                /* All ones, set output directly. */
                uint32_t index_start = current - in_begin;
                for (uint32_t i = 0; i < 32; i++) {
                    out_current[i] = index_start + i;
                }
                out_current += 32;
            }
        }
    }

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__one_or_two_bits(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros = _mm256_set1_epi8(0);

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i is_zero_byte_mask = _mm256_cmpeq_epi8(group, zeros);
        uint32_t is_zero_mask = _mm256_movemask_epi8(is_zero_byte_mask);
        if (is_zero_mask != 0xFFFFFFFF) {
            if (is_zero_mask != 0x00000000) {
                /* Group has zeros and ones. */
                uint32_t zero_amount = _mm_popcnt_u32(is_zero_mask);
                uint32_t is_non_zero_mask = ~is_zero_mask;
                uint32_t index = current - in_begin;

                if (zero_amount == 31) {
                    /* Only a single non-zero. */
                    uint32_t index_start = current - in_begin;
                    uint32_t index_offset = find_lowest_set_bit_index(
                        is_non_zero_mask);
                    *out_current = index_start + index_offset;
                    out_current++;
                }
                else if (zero_amount == 30) {
                    /* Exactly two non-zeros. */
                    uint32_t index_start = current - in_begin;
                    uint32_t index_offset1 = find_lowest_set_bit_index(
                        is_non_zero_mask);
                    uint32_t index_offset2 = find_highest_set_bit_index(
                        is_non_zero_mask);
                    out_current[0] = index_start + index_offset1;
                    out_current[1] = index_start + index_offset2;
                    out_current += 2;
                }
                else {
                    /* Branchless Algorithm */
                    for (uint32_t i = 0; i < 32; i++) {
                        *out_current = index;
                        out_current += is_non_zero_mask & 1;
                        index++;
                        is_non_zero_mask >>= 1;
                    }
                }
            }
            else {
                /* All ones, set output directly. */
                uint32_t index_start = current - in_begin;
                for (uint32_t i = 0; i < 32; i++) {
                    out_current[i] = index_start + i;
                }
                out_current += 32;
            }
        }
    }

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__counting_bits(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros = _mm256_set1_epi8(0);

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i is_zero_byte_mask = _mm256_cmpeq_epi8(group, zeros);
        uint32_t is_zero_mask = _mm256_movemask_epi8(is_zero_byte_mask);
        if (is_zero_mask != 0xFFFFFFFF) {
            if (is_zero_mask != 0x00000000) {
                /* Group has zeros and ones. */
                uint32_t zero_amount = _mm_popcnt_u32(is_zero_mask);
                uint32_t is_non_zero_mask = ~is_zero_mask;
                uint32_t index_start = current - in_begin;

                if (zero_amount == 31) {
                    /* Only a single non-zero. */
                    uint32_t index_offset = find_lowest_set_bit_index(
                        is_non_zero_mask);
                    *out_current = index_start + index_offset;
                    out_current++;
                }
                else if (zero_amount == 30) {
                    /* Exactly two non-zeros. */
                    uint32_t index_offset1 = find_lowest_set_bit_index(
                        is_non_zero_mask);
                    uint32_t index_offset2 = find_highest_set_bit_index(
                        is_non_zero_mask);
                    out_current[0] = index_start + index_offset1;
                    out_current[1] = index_start + index_offset2;
                    out_current += 2;
                }
                else {
                    /* More than two non-zeros. */
                    uint32_t index = index_start;
                    for (uint32_t i = 0; i < 32; i++) {
                        *out_current = index;
                        out_current += is_non_zero_mask & 1;
                        index++;
                        is_non_zero_mask >>= 1;
                    }
                }
            }
            else {
                /* All ones, set output directly. */
                uint32_t index_start = current - in_begin;
                for (uint32_t i = 0; i < 32; i++) {
                    out_current[i] = index_start + i;
                }
                out_current += 32;
            }
        }
    }

    return out_current - out_begin;
}

template<uint N>
static uint32_t find_non_zero_indices__bit_iteration(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros = _mm256_set1_epi8(0);

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i is_zero_byte_mask = _mm256_cmpeq_epi8(group, zeros);
        uint32_t is_zero_mask = _mm256_movemask_epi8(is_zero_byte_mask);
        if (is_zero_mask != 0xFFFFFFFF) {
            if (is_zero_mask != 0x00000000) {
                /* Group has zeros and ones. */
                uint32_t zero_amount = _mm_popcnt_u32(is_zero_mask);
                uint32_t is_non_zero_mask = ~is_zero_mask;
                uint32_t index_start = current - in_begin;

                if (zero_amount == 31) {
                    /* Only a single non-zero. */
                    uint32_t index_offset = find_lowest_set_bit_index(
                        is_non_zero_mask);
                    *out_current = index_start + index_offset;
                    out_current++;
                }
                else if (zero_amount == 30) {
                    /* Exactly two non-zeros. */
                    uint32_t index_offset1 = find_lowest_set_bit_index(
                        is_non_zero_mask);
                    uint32_t index_offset2 = find_highest_set_bit_index(
                        is_non_zero_mask);
                    out_current[0] = index_start + index_offset1;
                    out_current[1] = index_start + index_offset2;
                    out_current += 2;
                }
                else if (zero_amount > N) {
                    for (uint32_t i = 32; i > zero_amount; i--) {
                        uint index_offset = find_lowest_set_bit_index(
                            is_non_zero_mask);
                        *out_current = index_start = index_offset;
                        out_current++;
                        is_non_zero_mask &= ~(1 << index_offset);
                    }
                }
                else {
                    /* More than two non-zeros. */
                    uint32_t index = index_start;
                    for (uint32_t i = 0; i < 32; i++) {
                        *out_current = index;
                        out_current += is_non_zero_mask & 1;
                        index++;
                        is_non_zero_mask >>= 1;
                    }
                }
            }
            else {
                /* All ones, set output directly. */
                uint32_t index_start = current - in_begin;
                for (uint32_t i = 0; i < 32; i++) {
                    out_current[i] = index_start + i;
                }
                out_current += 32;
            }
        }
    }

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__grouped_2(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
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

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__grouped_4(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
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

    return out_current - out_begin;
}

static uint32_t find_non_zero_indices__grouped_8(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
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

    return out_current - out_begin;
}

// static uint32_t find_lowest_set_bit_index(uint32_t x)
// {
//     unsigned long index;
//     assert(_BitScanForward(&index, x));
//     return index;
// }

// static uint32_t find_highest_set_bit_index(uint32_t x)
// {
//     unsigned long index;
//     assert(_BitScanReverse(&index, x));
//     return index;
// }

// static uint32_t find_lowest_set_bit_index(uint64_t x)
// {
//     unsigned long index;
//     assert(_BitScanForward64(&index, x));
//     return index;
// }

// static uint32_t find_highest_set_bit_index(uint64_t x)
// {
//     unsigned long index;
//     assert(_BitScanReverse64(&index, x));
//     return index;
// }

// static uint32_t find_non_zero_indices__optimized(
//     uint8_t *__restrict in_begin,
//     uint8_t *in_end,
//     uint32_t *__restrict out_begin)
// {
//     assert((in_end - in_begin) % 32 == 0);

//     uint32_t *out_current = out_begin;

//     __m256i zeros;
//     __m256i ones;
//     memset(&zeros, 0x00, sizeof(__m256i));
//     memset(&ones, 0xFF, sizeof(__m256i));

//     for (uint8_t *current = in_begin; current != in_end; current += 32) {
//         __m256i group = _mm256_loadu_si256((__m256i *)current);
//         __m256i compared = _mm256_cmpeq_epi8(group, zeros);
//         compared = _mm256_andnot_si256(compared, ones);
//         uint32_t mask = _mm256_movemask_epi8(compared);
//         if (mask != 0) {
//             uint32_t set_bits = _mm_popcnt_u32(mask);
//             uint32_t index_offset = current - in_begin;
//             switch (set_bits) {
//                 case 1: {
//                     uint32_t index = find_lowest_set_bit_index(mask);
//                     *out_current = index_offset + index;
//                     out_current++;
//                     break;
//                 }
//                 case 2: {
//                     uint32_t index1 = find_lowest_set_bit_index(mask);
//                     uint32_t index2 = find_highest_set_bit_index(mask);
//                     *out_current = index_offset + index1;
//                     *(out_current + 1) = index_offset + index2;
//                     out_current += 2;
//                     break;
//                 }
//                 case 3:
//                 case 4:
//                 case 5:
//                 case 6:
//                 case 7:
//                 case 8:
//                 case 9:
//                 case 10:
//                 case 11:
//                 case 12:
//                 case 13:
//                 case 14:
//                 case 15:
//                 case 16:
//                 case 17:
//                 case 18:
//                 case 19:
//                 case 20:
//                 case 21:
//                 case 22:
//                 case 23:
//                 case 24:
//                 case 25:
//                 case 26:
//                 case 27:
//                 case 28:
//                 case 29:
//                 case 30:
//                 case 31: {
//                     uint32_t index_end = index_offset + 32;
//                     for (uint32_t index = index_offset; index < index_end;
//                          index++) {
//                         *out_current = index;
//                         bool is_non_zero = in_begin[index] != 0;
//                         out_current += is_non_zero;
//                     }
//                     break;
//                 }
//                 case 32: {
//                     __m256i index_offset_256 =
//                     _mm256_set1_epi32(index_offset);
//                     __m256i part1 = _mm256_add_epi32(
//                         index_offset_256,
//                         _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
//                     __m256i part2 = _mm256_add_epi32(
//                         index_offset_256,
//                         _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8));
//                     __m256i part3 = _mm256_add_epi32(
//                         index_offset_256,
//                         _mm256_set_epi32(23, 22, 21, 20, 19, 18, 17, 16));
//                     __m256i part4 = _mm256_add_epi32(
//                         index_offset_256,
//                         _mm256_set_epi32(31, 30, 29, 28, 27, 26, 25, 24));
//                     _mm256_store_si256((__m256i *)out_current, part1);
//                     _mm256_store_si256((__m256i *)out_current + 1, part2);
//                     _mm256_store_si256((__m256i *)out_current + 2, part3);
//                     _mm256_store_si256((__m256i *)out_current + 3, part4);
//                     out_current += 32;
//                     break;
//                 }
//             }
//         }
//     }

//     return out_current - out_begin;
// }

// static uint32_t find_non_zero_indices__optimized2(
//     uint8_t *__restrict in_begin,
//     uint8_t *in_end,
//     uint32_t *__restrict out_begin)
// {
//     assert((in_end - in_begin) % 64 == 0);

//     uint32_t *out_current = out_begin;

//     __m256i zeros;
//     memset(&zeros, 0x00, sizeof(__m256i));

//     for (uint8_t *current = in_begin; current != in_end; current += 64) {
//         __m256i group1 = _mm256_loadu_si256((__m256i *)current);
//         __m256i group2 = _mm256_loadu_si256((__m256i *)current + 1);
//         __m256i compared1 = _mm256_cmpeq_epi8(group1, zeros);
//         __m256i compared2 = _mm256_cmpeq_epi8(group2, zeros);
//         uint32_t inverted_mask1 = _mm256_movemask_epi8(compared1);
//         uint32_t inverted_mask2 = _mm256_movemask_epi8(compared2);
//         uint64_t inverted_mask = ((uint64_t)inverted_mask2 << (uint64_t)32)
//         |
//                                  inverted_mask1;
//         uint64_t mask = ~inverted_mask;
//         if (mask != 0) {
//             uint32_t set_bits = _mm_popcnt_u64(mask);
//             uint32_t start_index = current - in_begin;

//             if (set_bits < 16) {
//                 if (set_bits == 1) {
//                     uint32_t index = find_lowest_set_bit_index(mask);
//                     *out_current = start_index + index;
//                     out_current++;
//                 }
//                 else if (set_bits == 2) {
//                     uint32_t index1 = find_lowest_set_bit_index(mask);
//                     uint32_t index2 = find_highest_set_bit_index(mask);
//                     *out_current = start_index + index1;
//                     *(out_current + 1) = start_index + index2;
//                     out_current += 2;
//                 }
//                 else {
//                     unsigned long index_offset;
//                     while (_BitScanForward64(&index_offset, mask)) {
//                         *out_current = start_index + index_offset;
//                         out_current++;
//                         mask &= ~((uint64_t)1 << index_offset);
//                     }
//                 }
//             }
//             else if (set_bits < 64) {
//                 uint32_t index = start_index;
//                 while (mask) {
//                     *out_current = index;
//                     out_current += mask & 1;
//                     index++;
//                     mask >>= 1;
//                 }
//             }
//             else {
//                 __m256i index_offset_256 = _mm256_set1_epi32(start_index);

//                 __m256i part1 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
//                 __m256i part2 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8));
//                 __m256i part3 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(23, 22, 21, 20, 19, 18, 17, 16));
//                 __m256i part4 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(31, 30, 29, 28, 27, 26, 25, 24));

//                 __m256i part5 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(39, 38, 37, 36, 35, 34, 33, 32));
//                 __m256i part6 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(47, 46, 45, 44, 43, 42, 41, 40));
//                 __m256i part7 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(55, 54, 53, 52, 51, 50, 49, 48));
//                 __m256i part8 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(63, 62, 61, 60, 59, 58, 57, 56));

//                 _mm256_store_si256((__m256i *)out_current, part1);
//                 _mm256_store_si256((__m256i *)out_current + 1, part2);
//                 _mm256_store_si256((__m256i *)out_current + 2, part3);
//                 _mm256_store_si256((__m256i *)out_current + 3, part4);
//                 _mm256_store_si256((__m256i *)out_current + 4, part5);
//                 _mm256_store_si256((__m256i *)out_current + 5, part6);
//                 _mm256_store_si256((__m256i *)out_current + 6, part7);
//                 _mm256_store_si256((__m256i *)out_current + 7, part8);
//                 out_current += 64;
//             }
//         }
//     }

//     return out_current - out_begin;
// }

// static uint32_t find_non_zero_indices__optimized3(
//     uint8_t *__restrict in_begin,
//     uint8_t *in_end,
//     uint32_t *__restrict out_begin)
// {
//     assert((in_end - in_begin) % 64 == 0);

//     uint32_t *out_current = out_begin;

//     __m256i zeros;
//     memset(&zeros, 0x00, sizeof(__m256i));

//     for (uint8_t *current = in_begin; current != in_end; current += 64) {
//         __m256i group1 = _mm256_loadu_si256((__m256i *)current);
//         __m256i group2 = _mm256_loadu_si256((__m256i *)current + 1);
//         __m256i compared1 = _mm256_cmpeq_epi8(group1, zeros);
//         __m256i compared2 = _mm256_cmpeq_epi8(group2, zeros);
//         uint32_t inverted_mask1 = _mm256_movemask_epi8(compared1);
//         uint32_t inverted_mask2 = _mm256_movemask_epi8(compared2);
//         uint64_t inverted_mask = ((uint64_t)inverted_mask2 << (uint64_t)32)
//         |
//                                  inverted_mask1;
//         uint64_t mask = ~inverted_mask;
//         if (mask != 0) {
//             uint32_t set_bits = _mm_popcnt_u64(mask);
//             uint32_t start_index = current - in_begin;

//             if (set_bits < 16) {
//                 if (set_bits == 1) {
//                     uint32_t index = find_lowest_set_bit_index(mask);
//                     *out_current = start_index + index;
//                     out_current++;
//                 }
//                 else if (set_bits == 2) {
//                     uint32_t index1 = find_lowest_set_bit_index(mask);
//                     uint32_t index2 = find_highest_set_bit_index(mask);
//                     *out_current = start_index + index1;
//                     *(out_current + 1) = start_index + index2;
//                     out_current += 2;
//                 }
//                 else {
//                     unsigned long index_offset;
//                     while (_BitScanForward64(&index_offset, mask)) {
//                         *out_current = start_index + index_offset;
//                         out_current++;
//                         mask &= ~((uint64_t)1 << index_offset);
//                     }
//                 }
//             }
//             else if (set_bits < 64) {
//                 uint32_t index = start_index;
//                 while (mask) {
//                     *out_current = index;
//                     out_current += mask & 1;

//                     *out_current = index + 1;
//                     out_current += (mask & 2) >> 1;

//                     *out_current = index + 2;
//                     out_current += (mask & 4) >> 2;

//                     *out_current = index + 3;
//                     out_current += (mask & 8) >> 3;

//                     index += 4;
//                     mask >>= 4;
//                 }
//             }
//             else {
//                 __m256i index_offset_256 = _mm256_set1_epi32(start_index);

//                 __m256i part1 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
//                 __m256i part2 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8));
//                 __m256i part3 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(23, 22, 21, 20, 19, 18, 17, 16));
//                 __m256i part4 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(31, 30, 29, 28, 27, 26, 25, 24));

//                 __m256i part5 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(39, 38, 37, 36, 35, 34, 33, 32));
//                 __m256i part6 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(47, 46, 45, 44, 43, 42, 41, 40));
//                 __m256i part7 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(55, 54, 53, 52, 51, 50, 49, 48));
//                 __m256i part8 = _mm256_add_epi32(
//                     index_offset_256,
//                     _mm256_set_epi32(63, 62, 61, 60, 59, 58, 57, 56));

//                 _mm256_store_si256((__m256i *)out_current, part1);
//                 _mm256_store_si256((__m256i *)out_current + 1, part2);
//                 _mm256_store_si256((__m256i *)out_current + 2, part3);
//                 _mm256_store_si256((__m256i *)out_current + 3, part4);
//                 _mm256_store_si256((__m256i *)out_current + 4, part5);
//                 _mm256_store_si256((__m256i *)out_current + 5, part6);
//                 _mm256_store_si256((__m256i *)out_current + 6, part7);
//                 _mm256_store_si256((__m256i *)out_current + 7, part8);
//                 out_current += 64;
//             }
//         }
//     }

//     return out_current - out_begin;
// }

static uint32_t find_non_zero_indices__grouped_32(
    uint8_t *__restrict in_begin,
    uint8_t *in_end,
    uint32_t *__restrict out_begin)
{
    assert((in_end - in_begin) % 32 == 0);

    uint32_t *out_current = out_begin;

    __m256i zeros = _mm256_set1_epi8(0);

    for (uint8_t *current = in_begin; current != in_end; current += 32) {
        __m256i group = _mm256_loadu_si256((__m256i *)current);
        __m256i is_zero_byte_mask = _mm256_cmpeq_epi8(group, zeros);
        uint32_t is_zero_mask = _mm256_movemask_epi8(is_zero_byte_mask);
        uint32_t is_non_zero_mask = ~is_zero_mask;
        if (is_non_zero_mask != 0) {
            for (uint32_t i = 0; i < 32; i++) {
                if (current[i] != 0) {
                    *out_current++ = (current - in_begin) + i;
                }
            }
        }
    }

    return out_current - out_begin;
}

typedef uint32_t (*NonZeroFinder)(uint8_t *__restrict in_begin,
                                  uint8_t *in_end,
                                  uint32_t *__restrict out_begin);

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
    const uint32_t iteration_count = 20;
    const uint32_t cutoff = 3;

    std::vector<uint32_t> found(array.size());
    uint32_t amount_found = 0;
    std::vector<double> durations;

    for (uint32_t iteration = 0; iteration < iteration_count; iteration++) {
        TimePoint start_time = Clock::now();
        amount_found = function(
            array.data(), array.data() + array.size(), &found[0]);
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
                                         9'999'000,
                                         10'000'000};

    set_amounts = {};
    for (uint32_t i = 0; i <= 10'000'000; i += 100'000) {
        set_amounts.push_back(i);
    }

    // set_amounts = {10};

    std::vector<TestFunction> test_functions = {
        // {find_non_zero_indices__baseline, "Baseline"},
        // {find_non_zero_indices__grouped_2, "Groups of 2"},
        // {find_non_zero_indices__grouped_4, "Groups of 4"},
        // {find_non_zero_indices__grouped_8, "Groups of 8"},
        // {find_non_zero_indices__grouped_32, "Groups of 32"},
        {find_non_zero_indices__branch_free, "Branch Free"},
        // {find_non_zero_indices__grouped_branch_free, "Grouped Branchless"},
        {find_non_zero_indices__grouped_branch_free_2, "Grouped Branchless 2"},
        {find_non_zero_indices__mostly_ones, "Mostly Ones"},
        {find_non_zero_indices__one_or_two_bits, "One or Two Bits"},
        // {find_non_zero_indices__optimize_case_1_2, "One or Two Non-Zeros"},
        // {find_non_zero_indices__bit_iteration<0>, "Bit Iteration 0"},
        // {find_non_zero_indices__bit_iteration<4>, "Bit Iteration 4"},
        // {find_non_zero_indices__bit_iteration<7>, "Bit Iteration 7"},
        // {find_non_zero_indices__bit_iteration<10>, "Bit Iteration 10"},
        {find_non_zero_indices__bit_iteration<15>, "Bit Iteration 15"},
        // {find_non_zero_indices__bit_iteration<20>, "Bit Iteration 20"},
        // {find_non_zero_indices__bit_iteration<24>, "Bit Iteration 24"},
        // {find_non_zero_indices__bit_iteration<28>, "Bit Iteration 28"},
        // {find_non_zero_indices__optimized, "optimized"},
        // {find_non_zero_indices__optimized2, "optimized 2"},
        // {find_non_zero_indices__optimized3, "optimized 3"},
    };

    std::vector<uint32_t> all_indices(total_amount);
    for (uint32_t i = 0; i < total_amount; i++) {
        all_indices[i] = i;
    }
    std::random_shuffle(all_indices.begin(), all_indices.end());
    std::vector<uint8_t> data(total_amount);

    for (uint32_t set_amount : set_amounts) {

        std::fill(data.begin(), data.end(), 0);
        for (uint32_t i = 0; i < set_amount; i++) {
            data[all_indices[i]] = 1;
        }

        std::cout << "Set Amount: " << set_amount << "\n\n";

        for (auto test_function : test_functions) {
            ss << set_amount << ';';
            ss << test_function.name << ';';

            auto stats = run_test(
                test_function.name, test_function.function, data);

            ss << stats.min_ms << ';' << stats.max_ms << ';'
               << stats.average_ms << '\n';
        }

        std::cout << "\n\n";
    }

    std::ofstream file("benchmark_results.csv");
    file << ss.str();
    file.close();

    return 0;
}
