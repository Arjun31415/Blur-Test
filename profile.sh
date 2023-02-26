#!/bin/bash

cd build/ || exit
cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -j "$(nproc)"
rm "profile_results.txt" || true
rm "profile_results.csv" || true
rm "profile_multi_threaded.csv" || true
rm "profile_single_threaded.csv" || true
# loop over 1,3,5...kernel size
# for i in {3..50..2}; do
#     ./blur_test "$i" stressm
# done
for i in {3..50..2}; do
    ./blur_test "$i" stress
done
