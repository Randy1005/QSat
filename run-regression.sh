rm benchmarks/*.output
rm benchmarks/*.mem
rm inttests/*.csv
rm build/nohup.out

cd build
nohup ctest -L integration --timeout 2000 &
