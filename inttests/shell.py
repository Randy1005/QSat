#!/usr/bin/env python3
import os
import subprocess
import sys
import math
import time
from datetime import datetime
import pandas


# Usage: python shell.py [dimacs_cnf_file]


dirname = os.path.dirname
source_path = str(dirname(dirname(os.path.abspath(__file__))))

minisat_exe = source_path + "/build/bin/minisat"
qsat_exe = source_path + "/build/bin/QSat"
input_cnf = source_path + "/benchmarks/" + sys.argv[1]

# linux /bin/time executable
time_exe = "/bin/time"


csv_path = source_path + "/inttests/regression.csv"
first_row = False
if not os.path.exists(csv_path):
    first_row = True



minisat_solver_output  = input_cnf + ".minisat.output"
qsat_solver_output = input_cnf + ".qsat.output"
minisat_mem_output = input_cnf + ".minisat.mem"
qsat_mem_output = input_cnf + ".qsat.mem"


minisat_cmd = "/bin/time --format=\"%M\" " \
    + minisat_exe + " "\
    + input_cnf + " "\
    + minisat_solver_output\
    + " 2> " + minisat_mem_output

# we direct all outputs to devnull for now
start_time = time.time()
subprocess.run(minisat_cmd, 
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL) 
'''
subprocess.call([minisat_exe, input_cnf, minisat_solver_output], 
  stderr=subprocess.DEVNULL
)
'''
minisat_exec_time = time.time() - start_time;


qsat_cmd = "/bin/time --format=\"%M\" " \
    + qsat_exe + " "\
    + input_cnf + " "\
    + qsat_solver_output\
    + " 2> " + qsat_mem_output


start_time = time.time()
subprocess.run(qsat_cmd, 
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL) 
'''
subprocess.call([qsat_exe, input_cnf, qsat_solver_output],
  stdout=subprocess.DEVNULL,
  stderr=subprocess.DEVNULL
)
'''
qsat_exec_time = time.time() - start_time
time_diff = qsat_exec_time - minisat_exec_time

# compare outputs of minisat and qsat
# normally outputs contains only 2 lines:
# line 1 : SAT/UNSAT
# line 2 : if SAT, display solution models
# the solutions doesn't necessarily have to be the same
# but SAT/UNSAT must match
qsat_res = [line.strip() for line in open(qsat_solver_output)]
minisat_res = [line.strip() for line in open(minisat_solver_output)]

qsat_mem = float([line.strip() for line in open(qsat_mem_output)][0])
minisat_mem = float([line.strip() for line in open(minisat_mem_output)][1])


# compare SAT/UNSAT results
if qsat_res[0] != minisat_res[0]:
    #print("solver SAT/UNSAT mismatch!", file=sys.stderr)
    sys.exit(1)

# ignore the performance comparison for now
# enable this when we wanna know the difference
'''
if time_diff > 0 and time_diff / minisat_exec_time > 0.1:
    #print("qsat run time exceeds minisat runtime by more than 10%", file=sys.stderr)
    #print("difference(%) = " + str((time_diff / minisat_exec_time) * 100.0), file=sys.stderr)
    sys.exit(1)
'''


df = pandas.DataFrame([[sys.argv[1], 
    format(qsat_exec_time, '.4f'),
    format(minisat_exec_time, '.4f'),
    format(qsat_mem / 1000.0, '.2f'),
    format(minisat_mem / 1000.0, '.2f'),
    format(qsat_exec_time / minisat_exec_time, '.4f'),
    format(qsat_mem / minisat_mem, '.2f')]],
    columns=['test_case', 
        'qsat_runtime (sec)', 
        'minisat_runtime (sec)', 
        'qsat_mem (Mbyte)',
        'minisat_mem (Mbyte)',
        'runtime_slowdown (qsat/minisat)',
        'mem_usage_diff (qsat/minisat)'])

# df = pandas.concat(new_row, ignore_index=True)

if first_row:
    df.to_csv(csv_path, mode='a', index=False)
else:
    df.to_csv(csv_path, mode='a', index=False, header=False)

if os.path.isfile(minisat_solver_output):
    os.remove(minisat_solver_output)

if os.path.isfile(qsat_solver_output):
    os.remove(qsat_solver_output)

if os.path.isfile(minisat_mem_output):
    os.remove(minisat_mem_output)

if os.path.isfile(qsat_mem_output):
    os.remove(qsat_mem_output)
