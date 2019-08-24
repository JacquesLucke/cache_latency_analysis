#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <emmintrin.h>
#include <immintrin.h>
#include <assert.h>

#include "timeit.hpp"

static void find_indices__naive(
	uint8_t * __restrict in_begin, 
    uint8_t * __restrict in_end,
	uint32_t * __restrict out_begin, 
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

static void find_indices__grouped_2(
	uint8_t * __restrict in_begin, 
    uint8_t * __restrict in_end,
	uint32_t * __restrict out_begin, 
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

static void find_indices__grouped_4(
	uint8_t* __restrict in_begin,
	uint8_t* __restrict in_end,
	uint32_t* __restrict out_begin,
	uint32_t& out_amount)
{
	assert((in_end - in_begin) % 4 == 0);

	uint32_t* out_current = out_begin;

	for (uint8_t* current = in_begin; current != in_end; current += 4) {
		uint32_t group = *(uint32_t*)current;
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

static void find_indices__grouped_8(
	uint8_t* __restrict in_begin,
	uint8_t* __restrict in_end,
	uint32_t* __restrict out_begin,
	uint32_t& out_amount)
{
	assert((in_end - in_begin) % 8 == 0);

	uint32_t* out_current = out_begin;

	for (uint8_t* current = in_begin; current != in_end; current += 8) {
		uint64_t group = *(uint64_t*)current;
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

static void find_indices__grouped_32(
	uint8_t* __restrict in_begin,
	uint8_t* __restrict in_end,
	uint32_t* __restrict out_begin,
	uint32_t& out_amount)
{
	assert((in_end - in_begin) % 32 == 0);

	uint32_t* out_current = out_begin;

    __m256i zeros;
    __m256i ones;
    memset(&zeros, 0x00, sizeof(__m256i));
    memset(&ones, 0xFF, sizeof(__m256i));

	for (uint8_t* current = in_begin; current != in_end; current += 32) {
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

typedef void (*NonZeroFinder)(
    uint8_t* __restrict in_begin,
	uint8_t* __restrict in_end,
	uint32_t* __restrict out_begin,
	uint32_t& out_amount);

static void run_test(const char *name, NonZeroFinder function, std::vector<uint8_t> &array)
{
    std::vector<uint32_t> found(array.size());
    uint32_t amount_found = 0;

    for (uint32_t iteration = 0; iteration < 10; iteration++) {
		SCOPED_TIMER(name);
        function(array.data(), array.data() + array.size(), &found[0], amount_found);
	}

	std::cout << "Found " << amount_found << " non-zeros.\n";

	// std::cout << "Found:\n  ";
	// for (uint32_t i = 0; i < amount_found; i++) {
	// 	uint32_t index = found[i];
	// 	std::cout << index << " ";
	// }
    // std::cout << '\n';
}

int main(int argc, char const *argv[])
{
    uint32_t total_size = 10'000'000;
    uint32_t set_size = 100;

	std::vector<uint8_t> array(total_size);
    std::fill_n(array.begin(), array.size(), 0);

    std::random_device rd;
    std::uniform_int_distribution<uint32_t> distribution(0, total_size - 1);

    for (uint32_t i = 0; i < set_size; i++) {
        uint32_t index = distribution(rd);
        array[index] = 1;
    }

    run_test("naive", find_indices__naive, array);
    run_test("grouped 2", find_indices__grouped_2, array);
    run_test("grouped 4", find_indices__grouped_4, array);
    run_test("grouped 8", find_indices__grouped_8, array);
    run_test("grouped 32", find_indices__grouped_32, array);
    
    return 0;
}
