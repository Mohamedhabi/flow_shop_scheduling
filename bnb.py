
from fsp.branch_and_bound import *
from utils import Instance
import numpy as np

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

# u,v = johnson_partition(instance2)
# print(johnson_sort(u,1,False))
# print(johnson_sort(v,2,True))
# sequence = johnson_merge(u,v)
# finM1,startM2,finM2 = johnson_get_schedule(sequence)
# print(finM1)
# print(startM2)
# print(finM2)

# cost = finM2[-1][1]
result = get_results(instance2)
order = result["order"]
cost = result["C_max"]
print("Sequence === ",order)
print(f"Cost {cost}")
##print(f"Cost : {johnshon_calculateCost(instance)} time unit")
