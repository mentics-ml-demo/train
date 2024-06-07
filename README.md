# train
Continuous training of model as data arrives.

Optimizing dependencies only:
https://www.reddit.com/r/rust/comments/gvrgca/this_is_a_neat_trick_for_getting_good_runtime/

Profiling, get a flamegraph:
# install perf
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
# sudo apt-get remove linux-tools-common linux-tools-generic linux-tools-`uname -r`
# install flamegraph
cargo install flamegraph
# run executable with profiling
flamegraph --no-inline -o ignore/my_flamegraph.svg -- ../target/debug/train
flamegraph --no-inline -o ignore/my_flamegraph.svg -- ../target/release/train 20

to do profiling:
sudo sysctl kernel.perf_event_paranoid=-1
sudo bash -c "echo 0 > /proc/sys/kernel/kptr_restrict"

when done:
sudo sysctl kernel.perf_event_paranoid=4
sudo bash -c "echo 1 > /proc/sys/kernel/kptr_restrict"


cargo flamegraph --no-inline -o ignore/my_flamegraph.svg -- ../target/release/train
