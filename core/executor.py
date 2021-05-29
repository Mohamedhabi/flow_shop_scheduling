from utils import Instance
from core.methods import *
import fsp.parallel_bnb as parallel_bnb
import fsp.branch_and_bound as bnb
from fsp.branch_and_bound import *
import numpy as np
from time import perf_counter
from fsp import specific_heuristic_NEH, ACO, CDS, simulated_annealing, meta_heuristic_ga


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

def execute_others(instance : Instance, params, module, method):
    res = module.get_results(instance, **params)
    result =  {
        "instance_id": instance.id,
        "method": method,
        "makespan": int(res['C_max']),
        "execution_time": float(res['time']),
        "sequence": res['order'],
        "additional_info": params
    }
    print(result)
    return result, None

    
def execute(method_id: int, instance: Instance, params):
    if(method_id == BRANCH_AND_BOUND["id"]):
       return execute_bnb(instance,params)
    if(method_id == METHOD_NEH["id"]):
        return execute_others(instance, params, specific_heuristic_NEH, METHOD_NEH['method'])
    if(method_id == METHOD_ACO["id"]):
        return execute_others(instance, params, ACO, METHOD_ACO['method'])
    if(method_id == METHOD_CDS["id"]):
        return execute_others(instance, params, CDS, METHOD_CDS['method'])
    if(method_id == METHOD_SA["id"]):
        return execute_others(instance, params, simulated_annealing, METHOD_SA['method'])
    if(method_id == METHOD_GA["id"]):
        return execute_others(instance, params, meta_heuristic_ga, METHOD_GA['method'])