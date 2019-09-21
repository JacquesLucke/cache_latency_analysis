#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>

using uint = unsigned int;

struct ListItem {
    ListItem *next = nullptr;
    char pad[64 - sizeof(ListItem *)];
};

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Nanoseconds = std::chrono::nanoseconds;

uint do_not_optimize_value = 0;
template<class T> void do_not_optimize_away(T &&datum)
{
    do_not_optimize_value += (int)&datum;
}

__declspec(noinline) double measure_time_per_load(ListItem *begin)
{
    static volatile ListItem *s_final;

    uint loop_iterations = 1'000'000;
    uint total_loads = loop_iterations * 10;

    ListItem *current = begin;

    TimePoint start = Clock::now();

    for (uint i = 0; i < loop_iterations; i++) {
        current = current->next->next->next->next->next->next->next->next->next
                      ->next;
    }

    TimePoint end = Clock::now();

    do_not_optimize_away(current);

    Nanoseconds duration = end - start;
    double ns_per_load = (double)duration.count() / (double)total_loads;
    return ns_per_load;
}

double measure_time(std::vector<ListItem *> &randomized_items, uint amount)
{
    for (uint i = 0; i < amount; i++) {
        randomized_items[i]->next = randomized_items[(i + 1) % amount];
    }

    double duration_sum = 0;
    uint iterations = 100;
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
    for (uint i = 10; i < 1'000'000; i *= 1.2) {
        double time = measure_time(all_list_items, i);
        output_stream << "Measurement"
                      << ";" << i << ";" << time << '\n';
        std::cout << i << ": \t" << time << " ns\n";
    }

    std::ofstream file(
        "C:\\Users\\jacques\\Documents\\performance_tests\\benchmark_result."
        "csv");
    file << output_stream.str();
    file.close();

    std::cout << do_not_optimize_value << "\n";

    return 0;
}
