#!/usr/bin/env python3

# from pyutils import run
# import re
# run("rm tracemalloc.log")
# n = 8
# for x in range(n):
#     run(f"/usr/bin/time --verbose --output=ram{x}.txt ./playground.py -x {x}")
# trace = open("tracemalloc.log").readlines()
# trace = [int(re.sub("[()]","", t).split(",")[2]) for t in trace]
# print(trace)
# time = []
# for x in range(n):
#     with open(f"ram{x}.txt","r") as file:
#         for line in file:
#             if m:= re.search("Maximum.*?(\d+)", line):
#                 time.append(int(m.group(1)))
# print(time)
# for x in range(n):
#     print(f"trace/time = {round(trace[x]/time[x],5)},\ttime/trace = {round(time[x]/trace[x],5)}")
