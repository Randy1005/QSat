#!/usr/bin/env python3
import os
import subprocess
import sys
import math
import time
from datetime import datetime
import pandas
from openpyxl import Workbook
from openpyxl import load_workbook


# Usage: python shell.py [dimacs_cnf_file]

source_path = "/home/randy/QSat" 

minisat_exe = source_path + "/build/3rd-party/minisat/minisat"
qsat_exe = source_path + "/build/bin/QSat"
input_cnf = source_path + "/benchmarks/" + sys.argv[1]

csv_path = source_path + "/inttests/regression.csv"
first_row = False
if not os.path.exists(csv_path):
    first_row = True



minisat_output  = input_cnf + ".minisat.output"
qsat_output = input_cnf + ".qsat.output"

# we direct all outputs to devnull for now
start_time = time.time() 
subprocess.call([minisat_exe, input_cnf, minisat_output], 
  stdout=subprocess.DEVNULL,
  stderr=subprocess.DEVNULL
)
end_time = time.time();

minisat_exec_time = end_time - start_time;



start_time = time.time()
subprocess.call([qsat_exe, input_cnf, qsat_output],
  stdout=subprocess.DEVNULL,
  stderr=subprocess.DEVNULL
)
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



df = pandas.DataFrame()
new_series = pandas.Series({'test_case': sys.argv[1], 
    'qsat_runtime': format(qsat_exec_time, '.4f'), 
    'minisat_runtime': format(minisat_exec_time, '.4f'), 
    'slowdown': format(qsat_exec_time / minisat_exec_time, '.4f')}) 

df = df.append(new_series, ignore_index=True)

if first_row:
    df.to_csv(csv_path, mode='a', index=False)
else:
    df.to_csv(csv_path, mode='a', index=False, header=False)




if os.path.isfile(minisat_output):
    os.remove(minisat_output)

if os.path.isfile(qsat_output):
    os.remove(qsat_output)


