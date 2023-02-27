#!/bin/bash

cd build/ || exit
cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -j "$(nproc)"
# rm "profile_results.txt" || true
# rm "profile_results.csv" || true
# rm "profile_multi_threaded.csv" || true
# rm "profile_single_threaded.csv" || true
# loop over 1,3,5...kernel size
# for i in {3..50..2}; do
#     ./blur_test "$i" stressm
# done
# for i in {3..50..2}; do
#     ./blur_test "$i" stress
# done

for i in {3..101..2}; do
    nvprof -u us --log-file "./temp_profile.csv" --csv ./blur_test_cuda "$i" stress2d
    if [ "$i" -eq 3 ]; then
        tail -n +4 temp_profile.csv | head -n 2 >>profile_cuda_2d.csv
    fi
    tail -n +4 temp_profile.csv | grep "gaussian" >>profile_cuda_2d.csv
done

for i in {3..101..2}; do
    nvprof -u us --log-file "./temp_profile.csv" --csv ./blur_test_cuda "$i" stress1d
    if [ "$i" -eq 3 ]; then
        tail -n +4 temp_profile.csv | head -n 2 >>profile_cuda_1d.csv
    fi
    tail -n +4 temp_profile.csv | grep "gaussian" >>profile_cuda_1d.csv

done
