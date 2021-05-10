
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import numpy as np
from branch_and_bound import *
def get_results(bench: Benchmark):
    """Get the results of the algorithm on the specified instance

    Args:
        instance (class:instance): an FPS instance

    Returns:
        list: jobs order
    """

    return {
        "C_max": 13,
        "order":list(range(instance.get_jobs_number()))
        }
def get_cds_sub_sequences(instance: Instance):
    """an algorithem that create (m-1) 2-machines PFS instances from m-machine FPS instance

    Args:
        instance (class:instancdce): an FPS instance

    Returns:
        {
            "nb_sequences" : the number of 2-machines FPS instance created
            "sequences" : an array of nb_sequences FPS instances created
        }
    """
    m = instance.get_machines_number(); 
    n = instance.get_jobs_number(); 
    sequences = []; 
    for i in range(m-1):
        instance_two_machines = np.zeros((n,2)); 
        for j in range(i+1):
            instance_two_machines[:,0] += instance.get_machine_costs(j); 
            instance_two_machines[:,1] += instance.get_machine_costs(m-j-1); 
        sequences.append(Instance(instance_two_machines)); 
    return {
        "nb_sequences" : m-1,
        "sequences" : sequences
    }
def makespan(instance: Instance, schedule: list): 
    C_max = 0; # makespan 
    n = instance.get_jobs_number(); # number of jobs in the instance 
    m = instance.get_machines_number(); # number of machines 
    completionTimes = np.zeros((n,m)); #an (n,m) array that gives the completion time for a job i in machine j 
    #we iterate over the schedule list, foreach job, we calculate its completion time on all machines 
    for i in range(len(schedule)):
        jobCosts = instance.get_job_costs(schedule[i]); # an array of all operations' time for job i 
        if(i==0):
            completionTimes[schedule[i],:] = jobCosts; 
        else:
            for j in range(m):
                if(j == 0): #case when we are in the first operation for th i'th job 
                    completionTimes[schedule[i],j] = completionTimes[schedule[i-1],j] +jobCosts[j]; 
                else:
                    if(completionTimes[schedule[i],j-1] > completionTimes[schedule[i-1],j]):
                        completionTimes[schedule[i],j] = completionTimes[schedule[i],j-1]+jobCosts[j]; 
                    else: 
                        completionTimes[schedule[i],j] = completionTimes[schedule[i-1],j]+jobCosts[j]; 
    C_max = completionTimes[schedule[len(schedule)-1],m-1]; 
    return C_max; 

def cds(instance: Instance):
    """an Optimal O(m*nlogn) algorithm for solving the PFSP (n jobs, m machines) based on jhonson's algorithem

    Args:
        instance (class:instance): an FPS instance

    Returns:
        {
            "C_max" : the cost of the provided sequence 
            "order" : the order of jobs scheduled on the machines
        }
    """
    #creation of (m-1) virtual 2-machines PFS problem: 
    output = get_cds_sub_sequences(instance);
    m = output['nb_sequences'] + 1; 
    sequences = output['sequences']; 
    C_max = 0;  
    order = []; 
    for i in range(m-1): 
        #calculate the order using jhonson algorithem: 
        schedule = johnson(sequences[i]); 
        o = schedule['order']; 
        c = makespan(instance,o); 
        if(i==0):
            C_max = c; 
            order = o; 
        else:
            if(c<C_max):
                C_max = c; 
                order = o; 
    return {
        "C_max": C_max,
        "order": order
    } 
