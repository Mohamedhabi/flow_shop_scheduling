from utils import Instance
import numpy as np
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

def johnshon(instance : Instance):
    """an Optimal O(nlogn) algorithm for solving the FSP for the 2 machine case

    Args:
        instance (class:instance): an FPS instance

    Returns:
        {
            "C_max" : the cost of the provided sequence 
            "order" : the order of jobs scheduled on the machines
        }
    """

    U,V = johnson_partition(instance)
    U = johnson_sort(U,1,False)
    V = johnson_sort(V,2,True)
    M = johnson_merge(U,V)
    finM1,startM2,finM2 = johnson_get_schedule(M)
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
        return johnshon(instance)
    # if machine count is greater than 2, it is an NP-hard problem ==> apply branch and bound
    return {
        "C_max": 13,
        "order":list(range(instance.get_jobs_number()))
        }
