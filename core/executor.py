from utils import Instance
from core.methods import *
import fsp.parallel_bnb as parallel_bnb
import fsp.branch_and_bound as bnb
from fsp.branch_and_bound import *
import numpy as np
from time import perf_counter


def execute_bnb(instance : Instance, params : dict):
    rn = []
    exectime = -1
    makespan = -1
    addinfo = {}
    method=""
    print(f"executing {BRANCH_AND_BOUND['method']}")
    method = BRANCH_AND_BOUND['method']
    useHeuristique = params.get("use_heuristique",True)
    strategy = params.get("strategy_id",bnb.DEPTH_FIRST_SEARCH)
    parallel = params.get("parallel",False)
    if strategy > bnb.DEPTH_FIRST_SEARCH:
        strategy = bnb.DEPTH_FIRST_SEARCH
    t0 = perf_counter()
    result = None
    if(parallel):
        result,det= parallel_bnb.get_results(instance)
        exectime = perf_counter() - t0
        addinfo["leafs"] = det[2]
        addinfo["explored"] = det[0]
        addinfo["pruned"] = det[1]
    else:    
        result = bnb.get_results(instance,search_strategy=strategy,use_heuristique_init=useHeuristique)
        exectime = perf_counter() - t0
    rn = result["order"]
    makespan = result["C_max"]
    if not parallel:
        details = result.get("details", None)
        if(details is not None):
            addinfo["leafs"] = details["leafs"]
            addinfo["explored"] = details["explored"]
            addinfo["pruned"] = details["pruned"]
            addinfo["algorithm"] = BRANCH_AND_BOUND['method']
            exectime = details["time"]
        else:
            addinfo["algorithm"] = "johnson"
    print(result)
    res = {
        "instance_id": instance.id,
        "method": method,
        "makespan": int(makespan),
        "execution_time": exectime,
        "sequence": rn,
        "additional_info": addinfo
    }
    return res, None
    
def execute(method_id: int, instance: Instance, params: dict):
    rn = []
    exectime = -1
    makespan = -1
    addinfo = {}
    method=""
    if(method_id == BRANCH_AND_BOUND["id"]):
       return execute_bnb(instance,params)
    if(method_id == NEH["id"]):
        print(f"executing {NEH['method']}")
    if(method_id == CDS["id"]):
        print(f"executing {CDS['method']}")
    if(method_id == ACO["id"]):
        print(f"executing {ACO['method']}")
    if(method_id == SA["id"]):
        print(f"executing {SA['method']}")
    if(method_id == GA["id"]):
        print(f"executing {GA['method']}")

    print(rn)
    res = {
        "instance_id": instance.id,
        "method": method,
        "makespan": int(makespan),
        "execution_time": exectime,
        "sequence": rn,
        "additional_info": addinfo
    }
    return res, None
