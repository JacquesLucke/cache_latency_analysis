#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <nmmintrin.h>
#include <assert.h>

#include "timeit.hpp"

static void find_indices__naive(
	uint8_t* __restrict in_begin,
	uint8_t* __restrict in_end,
	uint32_t* __restrict out_begin,
	uint32_t& out_amount)
{
	uint32_t* out_current = out_begin;

	for (uint8_t* current = in_begin; current != in_end; current++) {
		if (*current != 0) {
			uint32_t index = current - in_begin;
			*out_current++ = index;
		}
	}

	out_amount = out_current - out_begin;
}

static void find_indices__branch_free(
	uint8_t* __restrict in_begin,
	uint8_t* __restrict in_end,
	uint32_t* __restrict out_begin,
	uint32_t& out_amount)
{
	uint32_t* out_current = out_begin;

	uint32_t amount = in_end - in_begin;
	for (uint32_t index = 0; index < amount; index++) {
		*out_current = index;
		uint8_t value = in_begin[index];
		bool is_non_zero = value != 0;
		out_current += is_non_zero;
	}

	out_amount = out_current - out_begin;
}

static void find_indices__grouped_2(
	uint8_t* __restrict in_begin,
	uint8_t* __restrict in_end,
	uint32_t* __restrict out_begin,
	uint32_t& out_amount)
{
	assert((in_end - in_begin) % 2 == 0);

	uint32_t* out_current = out_begin;

	for (uint8_t* current = in_begin; current != in_end; current += 2) {
		uint16_t group = *(uint16_t*)current;
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

static void find_indices__optimized(
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
		__m256i group = _mm256_loadu_si256((__m256i*)current);
		__m256i compared = _mm256_cmpeq_epi8(group, zeros);
		compared = _mm256_andnot_si256(compared, ones);
		uint32_t mask = _mm256_movemask_epi8(compared);
		if (mask != 0) {
			uint32_t set_bits = _mm_popcnt_u32(mask);
			uint32_t index_offset = current - in_begin;
			switch (set_bits) {
			case 1:
			{
				uint32_t index = find_lowest_set_bit_index(mask);
				*out_current = index_offset + index;
				out_current++;
				break;
			}
			case 2:
			{
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
			{
				uint32_t index_end = index_offset + 32;
				for (uint32_t index = index_offset; index < index_end; index++) {
					*out_current = index;
					bool is_non_zero = in_begin[index] != 0;
					out_current += is_non_zero;
				}
				break;
			}
			case 31:
			{
				uint32_t inverted_mask = ~mask;
				uint32_t zero_index = find_lowest_set_bit_index(inverted_mask) + index_offset;
				uint32_t end_index = index_offset + 32;
				for (uint32_t index = index_offset; index < zero_index; index++) {
					*out_current = index;
					out_current++;
				}
				for (uint32_t index = zero_index + 1; index < end_index; index++) {
					*out_current = index;
					out_current++;
				}
				break;
			}
			case 32:
				__m256i index_offset_256 = _mm256_set1_epi32(index_offset);
				__m256i part1 = _mm256_add_epi32(index_offset_256, _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
				__m256i part2 = _mm256_add_epi32(index_offset_256, _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8));
				__m256i part3 = _mm256_add_epi32(index_offset_256, _mm256_set_epi32(23, 22, 21, 20, 19, 18, 17, 16));
				__m256i part4 = _mm256_add_epi32(index_offset_256, _mm256_set_epi32(31, 30, 29, 28, 27, 26, 25, 24));
				_mm256_store_si256((__m256i*)out_current, part1);
				_mm256_store_si256((__m256i*)out_current + 1, part2);
				_mm256_store_si256((__m256i*)out_current + 2, part3);
				_mm256_store_si256((__m256i*)out_current + 3, part4);
				out_current += 32;
				break;
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
		__m256i group = _mm256_loadu_si256((__m256i*)current);
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

static void run_test(const char* name, NonZeroFinder function, std::vector<uint8_t>& array)
{
	std::vector<uint32_t> found(array.size());
	uint32_t amount_found = 0;

	for (uint32_t iteration = 0; iteration < 5; iteration++) {
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

int main(int argc, char const* argv[])
{
	uint32_t total_size = 10'000'000;
	uint32_t set_size = 7'000'000;

	std::cout << "Total size: " << total_size << "\n";

	std::vector<uint8_t> array(total_size);
	std::fill_n(array.begin(), array.size(), 0);

	std::random_device rd;
	std::uniform_int_distribution<uint32_t> distribution(0, total_size - 1);

	for (uint32_t i = 0; i < set_size; i++) {
		uint32_t index = distribution(rd);
		array[index] = 1;
	}

	//std::fill_n(array.begin(), array.size(), 1);

	run_test("naive", find_indices__naive, array);
	run_test("branch free", find_indices__branch_free, array);
	run_test("grouped 2", find_indices__grouped_2, array);
	run_test("grouped 4", find_indices__grouped_4, array);
	run_test("grouped 8", find_indices__grouped_8, array);
	run_test("grouped 32", find_indices__grouped_32, array);
	run_test("optimized", find_indices__optimized, array);

	return 0;
}
