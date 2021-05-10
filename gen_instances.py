from fsp import branch_and_bound
import matplotlib.pyplot as plt
from utils import Instance, Benchmark
import numpy as np
import codecs, json 
NB_INSTANCES_UNIF = 1
OUTPUT_FILE = "./benchmarks/instances_12_3.json"
JOBS = 12
MACHINES = 3
instances = []
# generating uniformly distributed costs instances
low = 10
high = 100
for i in range(NB_INSTANCES_UNIF):
    randomMat =np.round(np.random.uniform(low=low,high=high,size=(JOBS,MACHINES))) 
    randomMat = np.asarray(randomMat,dtype=np.int64)
    instance = Instance(randomMat)
    listrand = randomMat.tolist() # nested lists with same data, indices
    # solving
    result = branch_and_bound.get_results(instance,search_strategy=branch_and_bound.DEPTH_FIRST_SEARCH)
    instances.append({
        "jobs" : JOBS,
        "machines" : MACHINES,
        "distribution" : "uniform",
        "instance"  : listrand,
        "optimal_seq" : list(result["order"]),
        "makespan" : int(result["C_max"]) 
    })
# generating normal distributed costs instances
mean = 50
std = 50
NB_INSTANCES_NORM=1
for i in range(NB_INSTANCES_NORM):
    randomMat = np.round(np.abs(np.random.normal(loc=mean,scale=std,size=(JOBS,MACHINES))))
    randomMat = np.asarray(randomMat,dtype=np.int64)
    instance = Instance(randomMat)
    listrand = randomMat.tolist() # nested lists with same data, indices
     # solving
    result = branch_and_bound.get_results(instance,search_strategy=branch_and_bound.DEPTH_FIRST_SEARCH)
    instances.append({
        "jobs" : JOBS,
        "machines" : MACHINES,
        "distribution" : "normal",
        "instance"  : listrand,
        "optimal_seq" : list(result["order"]),
        "makespan" : int(result["C_max"]) 
    })

lamb = 100
NB_INSTANCES_EXP=1
for i in range(NB_INSTANCES_EXP):
    randomMat = np.round(np.random.exponential(scale=lamb,size=(JOBS,MACHINES)))
    randomMat = np.asarray(randomMat,dtype=np.int64)
    instance = Instance(randomMat)
    listrand = randomMat.tolist() # nested lists with same data, indices
     # solving
    result = branch_and_bound.get_results(instance,search_strategy=branch_and_bound.DEPTH_FIRST_SEARCH)
    instances.append({
        "jobs" : JOBS,
        "machines" : MACHINES,
        "distribution" : "exp",
        "instance"  : listrand,
        "optimal_seq" : list(result["order"]),
        "makespan" : int(result["C_max"]) 
    })

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError
with open(OUTPUT_FILE, 'w+') as outfile:
    json.dump(instances,outfile,default=convert)