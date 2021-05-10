from utils import Instance
import numpy as np
from branch_and_bound import *
def get_results(instance):
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
    return {
        "C_max": 13,
        "order":list(range(instance.get_jobs_number()))
        }
def get_cds_sub_sequences(instance: Instance)
    """an algorithem that create (m-1) 2-machines PFS instances from m-machine FPS instance

    Args:
        instance (class:instance): an FPS instance

    Returns:
        {
            "nb_sequences" : the number of 2-machines FPS instance created
            "sequences" : an array of nb_sequences FPS instances created
        }
    """
    m = instance.get_machines_number(); 
    n = instance.get_jobs_number(); 
    sequences = []; 
    for(i in range(m-1)):
        instance_two_machines = np.zeros(n,2); 
        for(j in range(i+1)):
            instance_two_machines[:,1] += instance.get_machine_costs(j+1); 
            instance_two_machines[:,2] += instance.get_machine_costs(m-j+1); 
        sequences.append(Instance(instance_two_machines)); 
    return {
        "nb_sequences" : m-1,
        "sequences" : sequences
    }; 

def cds(instance: Instance)
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
    m,sequences = get_cds_sub_sequences(instance); 
    C_max = 0; 
    order = [];
    for (i in range(m-1))
        #calculate the order using jhonson algorithem: 
        c , o = jhonson(sequences[i]);
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