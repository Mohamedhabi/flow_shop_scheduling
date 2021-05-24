import numpy as np
from utils import Instance
import fsp.branch_and_bound as bnb
import multiprocessing as mp
# params
BEST_FIRST_SEARCH = 0
DEPTH_FIRST_SEARCH = 1
LOCAL = 0
GLOBAL = 1
def get_results(instance : Instance,search_strategy=DEPTH_FIRST_SEARCH,log=False,scope=LOCAL):
    jobs = instance.get_jobs_number()
    machines = instance.get_machines_number()
    if machines == 2:
        return bnb.johnson(instance)

    nodes = createExecutionNodes(instance)    
    if scope == LOCAL:
        manager = mp.Manager()
        solutions = manager.dict()
        #solutions.items()
        parallelLocalBranchAndBound(nodes,instance,search_strategy,solutions,None,log)
        #print(solutions)
        return pickBestFromSolutions(solutions.items()),totalCounts(solutions.items())
    #print([(node.scheduled_jobs,node.eval) for node in nodes])
    #return solutions
def totalCounts(sols_dict):
    data = [0,0,0]
    for key,val in sols_dict:
        costs = val["costs"]
        data[0]+=costs[0]
        data[1]+=costs[1]
        data[2]+=costs[2]
    return data
def pickBestFromSolutions(sols_dict):
    Cmax = np.inf
    best = None
    for key,val in sols_dict:
        for sol in val["solutions"]:
            solcmax =sol["C_max"] 
            if  solcmax< Cmax:
                Cmax = solcmax
                best = sol["order"]
    return {"C_max" : Cmax,"order" : best}
def process_bnb_task(instance,nodes,level,upper_bound,process_id,sols_dict,counts):
    solutions = []
     # sequential 
    #print(upper_bound)
    costs = np.zeros(3)
    ub = upper_bound
    for node in nodes:
        currentbest,currentcost  = depthFirstSearchBranchAndBound(instance,node,level,ub,costs)
        if(currentcost < ub):
            ub = currentcost
        solutions.append({
            "order" : currentbest.scheduled_jobs,
            "C_max" : currentcost
        })
    print(solutions)
    sols_dict[process_id] = {
        "solutions" : solutions,
        "costs" : costs.tolist()
    }

def parallelLocalBranchAndBound(nodes : list,instance : Instance,search_strategy,solutions,counts,log):
    process_count = min(mp.cpu_count(),len(nodes))
    #chunk = int(np.floor(len(nodes)/process_count))
    processes = []
    # spliting chunks 
    i = 0
    lists = []
    for _ in range(process_count):
        lists.append([])
    for node in nodes:
        lists[i].append(node)
        i = (i +1) % process_count 
    for i in range(process_count):
        print(f"creating process {i}")
        process_nodes = lists[i]
        print(f"process {i} woring on : " + str([node.unscheduled_jobs for node in process_nodes]))
        p = mp.Process(target=process_bnb_task,args=(instance,process_nodes,1,np.inf,i,solutions,counts))
        processes.append(p)
        p.start()

    for i in range(process_count):
        processes[i].join()
    # sequential 
    # for node in nodes:
    #     currentbest,currentcost  = depthFirstSearchBranchAndBound(instance,node,1,np.inf)
    #     solutions.append({
    #         "order" : currentbest.scheduled_jobs,
    #         "C_max" : currentcost
    #     })
    #return solutions
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

def depthFirstSearchBranchAndBound(instance : Instance, node : bnb.Node,level,upper_bound,counts):
    counts[0] += 1 # explored
    machine_count = instance.get_machines_number()
    if(len(node.unscheduled_jobs) == 0):
        counts[2] += 1 # leaf
        cost = node.eval
        # if log : 
        #     print("leaf node : " + str(node.scheduled_jobs))
        #     print("leaf node cost: " + str(node.eval))
        # #count_array[2] += 1
        return node , cost
    # branching
    current_cost = upper_bound
    current_best= None
    for unsched_job in node.unscheduled_jobs:
        # create a new node
        newscheduled_jobs = list(node.scheduled_jobs)
        newscheduled_jobs.append(unsched_job)
        newunscheduled_jobs = set()
        # adding only the unscheduled jobs not selected 
        for u in node.unscheduled_jobs:
            if u is not unsched_job:
                newunscheduled_jobs.add(u)
        newnode = bnb.Node(newscheduled_jobs,newunscheduled_jobs,machine_count)
        newnode.calculateCost(instance,level,unsched_job,node.cost_array_row) 
        # if eval is greater or to than upper bound ==> dont add to nodelist (pruning the branch)
        # else add to an ordered list of nodes based on eval
        if newnode.eval < current_cost:
            best,cost = depthFirstSearchBranchAndBound(instance,newnode,(level+1),current_cost,counts)
            if(cost< current_cost):
                current_best = best
                current_cost = cost
        else:
            counts[1] += 1 # pruned
            #print("pruning")
        
    return current_best,current_cost