from multiprocessing.spawn import freeze_support
from fsp import parallel_bnb
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
#     [38 ,27 ,87 ,76 ,91 ,14, 29, 12, 77, 32, 87, 68, 94],
# [60  ,5 ,56  ,3 ,61 ,73 ,75 ,47 ,14 ,21 ,86 , 5 ,77],
# [23 ,57 ,64 , 7  ,1 ,63 ,41 ,63, 47 ,26 ,75 ,77, 40],
# [59 ,49 ,85 ,85  ,9 ,39, 41 ,56 ,40 ,54 ,77 ,51, 31],
# [41 ,69 ,13 ,86 ,72  ,8 ,49, 47 ,87, 58 ,18 ,68 ,28],
# ]).T
# benchmark_13_5
# instance = Instance(benchmark_13_5)
benchmark = JsonBenchmark(10,5)
ben = benchmark.get_instance_by_index(0)["instance"]
instance = Instance(np.array(ben))
#benchmark = Benchmark(20, 5, benchmark_folder = './benchmarks')
# random_mat = np.random.rand(9,3) * 100
# print("Instance :")

# randomInstance = Instance(
#     random_mat
# )
# print(str(randomInstance))
if __name__ == '__main__':
    tdfs1 = time.time() 
    result = parallel_bnb.get_results(instance,search_strategy=parallel_bnb.DEPTH_FIRST_SEARCH,log=False)
    tdfs2 = time.time()
    print(f"DFS took :{tdfs2 - tdfs1} s")
    print(result)
# tbfs1 = time.time() 
# result = parallel_bnb.get_results(instance4,search_strategy=parallel_bnb.BEST_FIRST_SEARCH,log=False)
# tbfs2 = time.time() 
# order = result["order"]
# cost = result["C_max"]
# print("Sequence === ",order)
# print(f"Cost {cost}")

# print(f"BFS took :{tbfs2 - tbfs1} s")

##print(f"Cost : {johnshon_calculateCost(instance)} time unit")