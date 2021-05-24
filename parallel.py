from multiprocessing.spawn import freeze_support
from fsp import parallel_bnb,branch_and_bound
from utils import Instance,Benchmark, JsonBenchmark
import numpy as np
import time
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
        [10,2,4,2],
        [1,4,5,1],
    ], dtype=np.int64)
)
# benchmark_13_5 = np.asarray([
#  [    61 ,75 ,26 ,62 ,62 ,11, 71 ,58 ,12 ,72 ,11 ,10 ,93],
#  [30 ,77, 22 ,30 ,88, 22, 91 ,15 ,42 ,12 ,19, 18 ,5],
#  [10 ,86 ,99 ,40 ,71 ,81 ,87 ,68 ,58 ,39 ,48 ,55 ,37],
#  [85 ,14 ,24 ,100 ,63 ,81 ,22 ,68 ,97 ,81, 76, 28 ,3],
#  [68 ,71 ,26 ,12 ,43 ,45 ,70 ,50 ,39 ,24 ,78 ,39 ,45],
# ]).T
# benchmark_13_5
# instance = Instance(benchmark_13_5)
benchmark = JsonBenchmark(6,5)
ben = benchmark.get_instance_by_index(1)["instance"]
instance = Instance(np.array(ben))
#benchmark = Benchmark(20, 5, benchmark_folder = './benchmarks')
# random_mat = np.random.rand(9,3) * 100
# print("Instance :")

# randomInstance = Instance(
#     random_mat
# )
# print(str(randomInstance))
time1=None
if __name__ == '__main__':
    tdfs1 = time.time() 
    result = parallel_bnb.get_results(instance,search_strategy=parallel_bnb.DEPTH_FIRST_SEARCH,log=False)
    tdfs2 = time.time()
    print(f"DFS took :{tdfs2 - tdfs1} s")
    print(result[0])
    print(f"explored {result[1][0]}")
    print(f"pruned {result[1][1]}")
    print(f"leafs {result[1][2]}")
    time1 = tdfs2 - tdfs1
# tbfs1 = time.time() 
# result = parallel_bnb.get_results(instance4,search_strategy=parallel_bnb.BEST_FIRST_SEARCH,log=False)
# tbfs2 = time.time() 
# order = result["order"]
# cost = result["C_max"]
# print("Sequence === ",order)
# print(f"Cost {cost}")

# print(f"BFS took :{tbfs2 - tbfs1} s")

##print(f"Cost : {johnshon_calculateCost(instance)} time unit")
jsonbenchmark = JsonBenchmark(6,5,benchmark_folder="./benchmarks")
instance = jsonbenchmark.get_instance_by_index(1)["instance"]
instance = Instance(np.asarray(instance))
time2 =None
if __name__ == '__main__':
    tdfs1 = time.time() 
    result = branch_and_bound.get_results(instance,search_strategy=branch_and_bound.DEPTH_FIRST_SEARCH,log=False)
    tdfs2 = time.time()
    print(f"DFS took :{tdfs2 - tdfs1} s")
    print(result)
    time2 = tdfs2 - tdfs1
    print(f"Acceleration {time2/time1}")

