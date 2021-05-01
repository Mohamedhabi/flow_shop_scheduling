import numpy as np
from utils import Instance
import fsp.branch_and_bound as bnb
import concurrent.futures
# params
BEST_FIRST_SEARCH = 0
DEPTH_FIRST_SEARCH = 1
LOCAL = 0
GLOBAL = 1
def get_results(instance : Instance,search_strategy=BEST_FIRST_SEARCH,log=False,scope=LOCAL):
    jobs = instance.get_jobs_number()
    machines = instance.get_machines_number()
    if machines == 2:
        return bnb.johnson(instance)

    nodes = createExecutionNodes(instance)    
    if scope == LOCAL:
        solutions = parallelLocalBranchAndBound(nodes,instance,search_strategy,log)
    
    #print([(node.scheduled_jobs,node.eval) for node in nodes])
    return {
        "C_max" : 0,
        "order" : None
    }

def parallelLocalBranchAndBound(nodes : list,instance : Instance,search_strategy,log):
    upper_bound = np.inf
    level = 1
    args = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for node in nodes:
            job = executor.submit(bnb.BandB,Args=[instance,level,node,upper_bound,None])
            print(job.result())

def createExecutionNodes(instance : Instance):
    job_count = instance.get_jobs_number();
    machine_count = instance.get_machines_number()
    nodes = []
    r = range(job_count)
    for job in r:
        unsched = []
        for x in r:
            if x is not job :
                unsched.append(x)
        node = bnb.Node([job],tuple(unsched),machine_count)
        # evaluating node
        node.calculateCost(instance,0,job,None)
        nodes.append(node)
    return nodes
    