# train
Continuous training of model as data arrives.

Optimizing dependencies only:
https://www.reddit.com/r/rust/comments/gvrgca/this_is_a_neat_trick_for_getting_good_runtime/

Profiling, get a flamegraph:
# install perf
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
# install flamegraph
cargo install flamegraph
# run executable with profiling
flamegraph --no-inline -o ignore/my_flamegraph.svg -- ../target/debug/train