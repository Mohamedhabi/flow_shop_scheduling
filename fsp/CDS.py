
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import numpy as np
from .branch_and_bound import *
import time

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
    m = instance.get_machines_number()
    n = instance.get_jobs_number()
    sequences = []
    for i in range(m-1):
        instance_two_machines = np.zeros((n,2))
        for j in range(i+1):
            instance_two_machines[:,0] += instance.get_machine_costs(j)
            instance_two_machines[:,1] += instance.get_machine_costs(m-j-1)
        sequences.append(Instance(instance_two_machines))
    return {
        "nb_sequences" : m-1,
        "sequences" : sequences
    }

def get_results(instance: Instance):
    """an Optimal O(m*nlogn) algorithm for solving the PFSP (n jobs, m machines) based on jhonson's algorithem

    Args:
        instance (class:instance): an FPS instance

    Returns:
        {
            "C_max" : the cost of the provided sequence 
            "order" : the order of jobs scheduled on the machines
        }
    """
    start = time.perf_counter()
    #creation of (m-1) virtual 2-machines PFS problem: 
    output = get_cds_sub_sequences(instance)
    m = output['nb_sequences'] + 1
    sequences = output['sequences'] 
    C_max = 0
    order = []
    for i in range(m-1): 
        #calculate the order using jhonson algorithem: 
        schedule = johnson(sequences[i])
        o = schedule['order']
        c = instance.makespan(o)
        if(i==0):
            C_max = c 
            order = o 
        else:
            if(c < C_max):
                C_max = c 
                order = o 
    return {
        "C_max" :  C_max,
        "order" : order,
        "time" : time.perf_counter() - start,
    }