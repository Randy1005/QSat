#!/usr/bin/env python3
import os
import subprocess
import sys
import math
import time
from datetime import datetime
import pandas
import psutil

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

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


timeout_lim = 600

minisat_solver_output  = input_cnf + ".minisat.output"
qsat_solver_output = input_cnf + ".qsat.output"
minisat_mem_output = input_cnf + ".minisat.mem"
qsat_mem_output = input_cnf + ".qsat.mem"

minisat_timedout = False
minisat_cmd = "/bin/time --format=\"%M\" " \
    + minisat_exe + " "\
    + input_cnf + " "\
    + minisat_solver_output\
    + " 2> " + minisat_mem_output

# we direct all outputs to devnull for now
start_time = time.time()
process = subprocess.Popen("exec " + minisat_cmd, 
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL) 

try:
    process.wait(timeout=timeout_lim)
except subprocess.TimeoutExpired:
    kill(process.pid)
    minisat_timedout = True
minisat_exec_time = time.time() - start_time;


qsat_timedout = False
qsat_cmd = "/bin/time --format=\"%M\" " \
    + qsat_exe + " "\
    + input_cnf + " "\
    + qsat_solver_output\
    + " 2> " + qsat_mem_output


start_time = time.time()
process2 = subprocess.Popen("exec " + qsat_cmd, 
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL) 

try:
    process2.wait(timeout=timeout_lim)
except subprocess.TimeoutExpired:
    kill(process2.pid)
    qsat_timedout = True
qsat_exec_time = time.time() - start_time

if not qsat_timedout:
    qsat_res = [line.strip() for line in open(qsat_solver_output)]
    qsat_mem = float([line.strip() for line in open(qsat_mem_output)][0])

if not minisat_timedout:
    minisat_res = [line.strip() for line in open(minisat_solver_output)]
    minisat_mem = float([line.strip() for line in open(minisat_mem_output)][1])

# ignore the performance comparison for now

qsat_exec_time = format(qsat_exec_time, '.4f') if not qsat_timedout else ("> " + str(timeout_lim) + "(timed out)") 
minisat_exec_time = format(minisat_exec_time, '.4f') if not minisat_timedout else ("> " + str(timeout_lim) + "(timed out)") 
qsat_mem = format(qsat_mem / 1000.0, '.2f') if not qsat_timedout else "N/A (timed out)" 
minisat_mem = format(minisat_mem / 1000.0, '.2f') if not minisat_timedout else "N/A (timed out)" 
slowdown = format(float(qsat_exec_time) / float(minisat_exec_time), '.2f') if not minisat_timedout and not qsat_timedout else "N/A (timed out)"
mem_diff = format(float(qsat_mem) / float(minisat_mem), '.2f') if not minisat_timedout and not qsat_timedout else "N/A (timed out)"
df = pandas.DataFrame([[sys.argv[1], 
    qsat_exec_time,
    minisat_exec_time,
    qsat_mem,
    minisat_mem,
    slowdown,
    mem_diff]],
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


# compare outputs of minisat and qsat
# normally outputs contains only 2 lines:
# line 1 : SAT/UNSAT
# line 2 : if SAT, display solution models
# the solutions doesn't necessarily have to be the same
# but SAT/UNSAT must match

# compare SAT/UNSAT results
if not qsat_timedout and not minisat_timedout:
    if qsat_res[0] != minisat_res[0]:
        #print("solver SAT/UNSAT mismatch!", file=sys.stderr)
        sys.exit(1)

