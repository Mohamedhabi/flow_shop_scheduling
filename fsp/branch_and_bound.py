from utils import Instance
import numpy as np
from sortedcontainers import SortedList
def johnson_get_schedule(sequence : list):
    """
    Returns the scheduling of jobs namely : finish dates on M1, start dates on M2 , finish dates on M2
    """
    acc = 0 # accumulates processing time on m1
    dates1=[] # dates of finish on m1
    dates2 = [] # dates of start on m2
    dates3 = [] # dates of finish on m2
    idx = 0
    prevJob = None # saves the previous job
    for job in sequence:
        index,cost1,cost2 = job
        acc = acc + job[1]
        dates1.append((job[0],acc)) # append (job_index, finisht_time_m1)
        if idx == 0: 
            ## the first element always starts on m2 after it finishes on m1
            dates2.append((index,cost1))
            dates3.append((index,cost1+cost2))
        else:
            # not the first scheduled job we get the previous job
            indexprev,cost1prev,cost2prev = prevJob
            # start date will be the max between start of prev job + its cost and the finish date of this job on M1
            date = max((dates2[idx-1][1] + cost2prev), dates1[idx][1]) 
            dates2.append((index,date))
            # the finish date of this job is its start date + its cost
            dates3.append((index,date+cost2))
        
        prevJob = job
        idx+=1
    return dates1,dates2,dates3
    

def johnson_partition(instance: Instance):
    job_count = instance.get_jobs_number()
    ulist = []
    vlist = []
    for i in range(job_count):
        costm1 = instance.get_cost(i,0)
        costm2 = instance.get_cost(i,1)
        if costm1 < costm2 :
            ulist.append((i,costm1,costm2))
        else:
            vlist.append((i,costm1,costm2))
    return ulist,vlist


def johnson_sort(list : list,index,desc):
    list.sort(key=lambda x: x[index],reverse=desc)
    return list
def johnson_merge(list1: list,list2:list):
    list1.extend(list2)
    return list1

def johnson(instance : Instance):
    """an Optimal O(nlogn) algorithm for solving the FSP for the 2 machine case

    Args:
        instance (class:instance): an FPS instance

    Returns:
        {
            "C_max" : the cost of the provided sequence 
            "order" : the order of jobs scheduled on the machines
        }
    """
    print("Johnshon optimal algorithm on 2-machine Instance : ")
    U,V = johnson_partition(instance)
    print("U : " +str(U))
    print("V : " +str(V))
    U = johnson_sort(U,1,False)
    V = johnson_sort(V,2,True)
    print("U sorted : " +str(U))
    print("V sorted : " +str(V))
    
    M = johnson_merge(U,V)
    print("Sequence : " +str(M))
    
    finM1,startM2,finM2 = johnson_get_schedule(M)
    print("Schedule: ")
    print("End on M1 " + str(finM1))
    print("Start on M2 " + str(startM2))
    print("End on M2 " + str(finM2))
    ## return values
    cmax = finM2[-1][1] # date of finish of the last scheduled job on M2 
    order = [ job[0]  for job in M]
    return {
        "C_max" : cmax,
        "order" : order  
    }

def get_results(instance : Instance):
    """Get the results of the algorithm on the specified instance

    Args:
        instance (class:instance): an FPS instance

    Returns:
        list: jobs order
    """

    #exemples
    instance.get_cost(0,0) # The cost of Job 0, on machine 0
    instance.get_job_costs(0) # the costs of job 0 on all machines
    instance.get_machine_costs(0) # Get the costs of all jobs on machine 0
    #...
    machine_count = instance.get_machines_number()
    if machine_count == 2:
        return johnson(instance)
    # if machine count is greater than 2, it is an NP-hard problem ==> apply branch and bound
    return general_case_branch_and_bound(instance)
class Node:
     def __init__(self,scheduled,unscheduled,machine_count):
        self.eval = 0
        self.cost_array_row = np.zeros((machine_count))
        self.scheduled_jobs = list(scheduled)
        self.unscheduled_jobs = set(unscheduled)
     
     def calculateCost(self,instance,level,job_index,prior_cost_row):
         machine = self.cost_array_row.size # nb machines
         for mach in range (machine):
             top = 0 if level == 0 else prior_cost_row[mach]
             left = 0 if mach == 0 else self.cost_array_row[mach-1]
             self.cost_array_row[mach] = max(top,left) + instance.get_cost(job_index,mach)
         # the evaluation is the last value calculated
         self.eval = self.cost_array_row[machine-1]
  
def general_case_branch_and_bound(instance :Instance):
    machine_count = instance.get_machines_number()
    jobs_count = instance.get_jobs_number()
    print("machine count : " + str(machine_count))
    print("jobs count : " + str(jobs_count))
    starting_seq = tuple(range(jobs_count))
    starting_node = Node([],starting_seq,machine_count)
    starting_node.eval = np.inf
    upper_bound = np.inf
    level = 0
    count_dict = {
        "explored" : 0,
        "pruned" : 0,
        "leaf" : 0
    }
    bestNode,cost = BandB(instance,level,starting_node,upper_bound,count_dict)

    print("Nodes explored : " + str(count_dict["explored"]))
    
    print("Nodes pruned : " + str(count_dict["pruned"]))
    
    print("Leafs reached : " + str(count_dict["leaf"]))
    
    return {
        "C_max" :  cost,
        "order" : bestNode.scheduled_jobs
    }

def BandB(instance: Instance,level : int ,node: Node,upper_bound: np.float,count_dict: dict):
    machine_count = instance.get_machines_number()
    count_dict["explored"] += 1
    print("exploring node: " + str(node.scheduled_jobs) + "/" + str(node.unscheduled_jobs))
    # this is a leef node can't be branched, we only calculate cost of this node
    ub = upper_bound
    print("upper bound for cost is : " + str(ub))
    if(len(node.unscheduled_jobs) == 0):
        cost = node.eval
        print("leaf node : " + str(node.scheduled_jobs))
        print("leaf node cost: " + str(node.eval))
        count_dict["leaf"] += 1
        return node , cost#,explored
    next_nodelist = SortedList(key=lambda x: x.eval) # using a sorted list for a Best first Search strategy
    # branching
    for unsched_job in node.unscheduled_jobs:
        # create a new node
        newscheduled_jobs = list(node.scheduled_jobs)
        newscheduled_jobs.append(unsched_job)
        newunscheduled_jobs = set()
        # adding only the unscheduled jobs not selected 
        for u in node.unscheduled_jobs:
            if u is not unsched_job:
                newunscheduled_jobs.add(u)
        
        newnode = Node(newscheduled_jobs,newunscheduled_jobs,machine_count)
        print("node branched: " + str(newnode.scheduled_jobs) + "/" + str(newnode.unscheduled_jobs))
    
        # evalute node (bounding)
        newnode.calculateCost(instance,level,unsched_job,node.cost_array_row)
        print("node : " + str(newnode.scheduled_jobs) + "/" + str(newnode.unscheduled_jobs) +" cost : " +str(newnode.eval))
    
        # if eval is greater or to than lower_bound ==> dont add to nodelist (pruning the branch)
        # else add to an ordered list of nodes based on eval
        if newnode.eval < ub:
            next_nodelist.add(newnode)
            print("adding node : " + str(newnode.scheduled_jobs) + "/" + str(newnode.unscheduled_jobs))
        # we prune only when it is not a leaf node
        elif len(newnode.scheduled_jobs) < instance.get_jobs_number():
            print("pruning node : " + str(newnode.scheduled_jobs) + "/" + str(newnode.unscheduled_jobs))
            count_dict["pruned"] += 1
    # Best first search based on eval
    currentBest = node
    currentCost= ub
    # exploring the tree of nodes
    lvl = level +1 # current level of tree
    for next_node in next_nodelist:
        # if the lowerbound is updated we must recheck the node if we need to explore it
        if(next_node.eval <= ub):
            #explored +=1
            best, cost = BandB(instance,lvl,next_node,ub,count_dict)
            #explored += exp
            if cost < currentCost:
                currentBest = best
                currentCost = cost
                ub = cost # updating lower bound to prune branches
        else:
            count_dict["pruned"] += 1
            print("pruning node : " + str(newnode.scheduled_jobs) + "/" + str(newnode.unscheduled_jobs))
    if(len(currentBest.scheduled_jobs) == instance.get_jobs_number()) :
        print("best branch node : " + str(currentBest.scheduled_jobs))
        print("best branch node cost (lower bound): " + str(currentBest.eval))
        
    return currentBest,currentCost#,explored

        

