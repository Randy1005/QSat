# not sure if this is the correct way to remove bad characters
# so far the sat benchmarks have "%" and a single "0" in one line
# "%" would cause crash, and "0" would represent a empty clause
# minisat would directly return UNSAT if formulas contain empty clauses

import sys
with open(sys.argv[1], "r") as f:
    lines = f.readlines()
with open(sys.argv[1], "w") as f:
    for line in lines:
        if len(line.strip("\n")) > 1 and not("%" in line):
            f.write(line)
