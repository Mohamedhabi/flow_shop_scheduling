
from fsp.branch_and_bound import *
from utils import Instance,Benchmark
import numpy as np
#import argparse
#parser = argparse.ArgumentParser(description='Exact methods flow shop permutation problem')
instance1 = Instance(
    np.array([
        [2, 1],
        [5, 7],
        [6, 1],
        [4, 2]
    ], dtype=np.int64)
)
instance2 = Instance(
    np.array([
        [3, 5],
        [4, 2],
        [6, 1],
        [3, 3]
    ], dtype=np.int64)
)
instance3 = Instance(
    np.array([
        [3,2,3],
        [1,4,2],
        [3,2,1],
    ], dtype=np.int64)
)

instance4 = Instance(
    np.array([
        [1,2,3,2],
        [1,4,2,10],
        [3,2,1,5],
        [4,10,3,1],
        [1,5,4,4],
        [2,3,2,6],
        [5,2,1,1],
    ], dtype=np.int64)
)

#benchmark = Benchmark(20, 5, benchmark_folder = './benchmarks')
random_mat = np.random.rand(9,3) * 100
print("Instance :")
print(random_mat)
randomInstance = Instance(
    random_mat
)
result = get_results(randomInstance)
order = result["order"]
cost = result["C_max"]
print("Sequence === ",order)
print(f"Cost {cost}")
##print(f"Cost : {johnshon_calculateCost(instance)} time unit")
