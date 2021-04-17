import numpy as np
from utils import Instance
import fsp.brute_force as brute

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
        [1,2,3,2],
        [1,4,2,10],
        [3,2,1,5],
        [4,10,3,1],
        [1,5,4,4],
        [2,3,2,6],
        [5,2,1,1],
    ])
)

result = brute.get_results(instance3)
order = result["order"]
cost = result["C_max"]
print("Sequence === ",order)
print(f"Cost {cost}")