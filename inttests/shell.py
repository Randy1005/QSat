#!/usr/bin/env python3
import os
import subprocess
import sys
import math
import time


# Usage: python shell.py [dimacs_cnf_file]

source_path = "/home/randy/QSat" 

minisat_exe = source_path + "/build/3rd-party/minisat/minisat"
qsat_exe = source_path + "/build/main/QSat"
input_cnf = source_path + "/benchmarks/" + sys.argv[1]



minisat_output  = input_cnf + ".minisat.output"
qsat_output = input_cnf + ".qsat.output"


start_time = time.time() 
subprocess.call([minisat_exe, input_cnf, minisat_output])
end_time = time.time();

minisat_exec_time = end_time - start_time;

start_time = time.time()
subprocess.call([qsat_exe, input_cnf, qsat_output])
end_time = time.time()

qsat_exec_time = end_time - start_time
time_diff = qsat_exec_time - minisat_exec_time

# compare outputs of minisat and qsat
# normally outputs contains only 2 lines:
# line 1 : SAT/UNSAT
# line 2 : if SAT, display solution models
# the solutions doesn't necessarily have to be the same
# but SAT/UNSAT must match
qsat_res = [line.strip() for line in open(qsat_output)]
minisat_res = [line.strip() for line in open(minisat_output)]



# compare SAT/UNSAT results
if qsat_res[0] != minisat_res[0]:
    print("solver SAT/UNSAT mismatch!", file=sys.stderr)
    sys.exit(1)

if time_diff > 0 and time_diff / minisat_exec_time > 0.1:
    print("qsat run time exceeds minisat runtime by more than 10%", file=sys.stderr)
    print("difference(%) = " + str((time_diff / minisat_exec_time) * 100.0), file=sys.stderr)
    sys.exit(1)

if os.path.isfile(minisat_output):
    os.remove(minisat_output)

if os.path.isfile(qsat_output):
    os.remove(qsat_output)

