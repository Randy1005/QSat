rm benchmarks/*.output
rm benchmarks/*.mem
rm inttests/*.csv

cd build
nohup ctest --timeout 2000 & > regression.out
