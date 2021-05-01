
from fsp.branch_and_bound import *
from utils import Instance,Benchmark
import numpy as np
import time

def exec(instance,strategy = BEST_FIRST_SEARCH):
    t1 = time.time() 
    get_results(instance,search_strategy=DEPTH_FIRST_SEARCH,log=False)
    t2 = time.time()
    return t2-t1
bfs_list = []
dfs_list = []

for i in range(3):
    random_mat = np.random.rand(9,3) * 100
    randomInstance = Instance(
        random_mat
    )
    tbfs = exec(randomInstance,strategy=BEST_FIRST_SEARCH)
    bfs_list.append(tbfs)
    tdfs = exec(randomInstance,strategy=DEPTH_FIRST_SEARCH)
    dfs_list.append(tdfs)

avg_bfs = np.asarray(bfs_list).mean()

avg_dfs = np.asarray(dfs_list).mean()
print("Average BFS time : " + str(avg_bfs))

print("Average DFS time : " + str(avg_dfs))

accel = avg_bfs / avg_dfs
print("BFS / DFS : " + str(accel))