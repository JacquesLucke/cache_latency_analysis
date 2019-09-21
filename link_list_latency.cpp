#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>
#include <xmmintrin.h>

using uint = unsigned int;

struct ListItem {
    ListItem *next = nullptr;
    ListItem *prefetch = nullptr;
    char pad[64 - 2 * sizeof(ListItem *)];
};

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Nanoseconds = std::chrono::nanoseconds;

uint do_not_optimize_value = 0;
template<class T> void do_not_optimize_away(T &&datum)
{
    do_not_optimize_value += (int)&datum;
}

double measure_time_per_load(ListItem *begin)
{
    static volatile ListItem *s_final;

    uint loop_iterations = 1'000'000;
    uint total_loads = loop_iterations * 10;

    ListItem *current = begin;

    TimePoint start = Clock::now();

    for (uint i = 0; i < loop_iterations; i++) {
        for (uint j = 0; j < 10; j++) {
            current = current->next;
            _mm_prefetch((char *)current->prefetch, 0);
        }
    }

    TimePoint end = Clock::now();

    do_not_optimize_away(current);

    Nanoseconds duration = end - start;
    double ns_per_load = (double)duration.count() / (double)total_loads;
    return ns_per_load;
}

double measure_time(std::vector<ListItem *> &randomized_items,
                    uint amount,
                    uint prefetch_distance)
{
    for (uint i = 0; i < amount; i++) {
        randomized_items[i]->next = randomized_items[(i + 1) % amount];
        randomized_items[i]->prefetch =
            randomized_items[(i + prefetch_distance) % amount];
    }

    double duration_sum = 0;
    uint iterations = 20;
    for (uint i = 0; i < iterations; i++) {
        duration_sum += measure_time_per_load(randomized_items[0]);
    }
    double average_duration = duration_sum / (double)iterations;
    return average_duration;
}

int main(int argc, char const *argv[])
{
    std::cout << sizeof(ListItem) << '\n';
    uint allocation_amount = 10'000'000;

    std::vector<ListItem *> all_list_items;
    all_list_items.reserve(allocation_amount);

    for (uint i = 0; i < allocation_amount; i++) {
        all_list_items.push_back(new ListItem());
    }

    std::shuffle(all_list_items.begin(),
                 all_list_items.end(),
                 std::default_random_engine());

    std::stringstream output_stream;
    for (uint prefetch_distance = 0; prefetch_distance < 20;
         prefetch_distance += 1) {
        for (uint i = 10; i < 1'000'000; i *= 1.3) {
            double time = measure_time(all_list_items, i, prefetch_distance);
            output_stream << "Prefetch " << prefetch_distance << ";" << i
                          << ";" << time << '\n';
            std::cout << "Prefetch " << prefetch_distance << " - " << i
                      << ": \t" << time << " ns\n";
        }
    }

    std::ofstream file(
        "C:\\Users\\jacques\\Documents\\performance_tests\\benchmark_result."
        "csv");
    file << output_stream.str();
    file.close();

    std::cout << do_not_optimize_value << "\n";

    return 0;
}
