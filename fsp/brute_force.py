from utils import Instance
import numpy as np
import itertools as iter
def get_sequences(instance : Instance):
    jobs_count = instance.get_jobs_number()
    return iter.permutations(range(jobs_count))
def populate_costs(sequence : tuple,cost_array : np.ndarray , instance : Instance):
    #jobs_count = instance.get_jobs_number()
    machine_count = instance.get_machines_number()
    job_index = 0
    for job in sequence:
        for machine in range(machine_count):
            cost = instance.get_cost(job,machine)
            top = 0 if job_index == 0 else cost_array[job_index-1][machine]
            left = 0 if machine == 0 else cost_array[job_index][machine-1] 
            cost_array[job_index][machine] = max(top,left) + cost
        job_index += 1


def brute_force(instance : Instance):
    jobs_count = instance.get_jobs_number()
    machine_count = instance.get_machines_number()
    cost_array = np.zeros((jobs_count,machine_count),dtype=np.int32)
    best_sequence = None
    Cmax = np.inf
    # Complexity of O(n *m * n!)
    for sequence in get_sequences(instance): # get_sequences is O(n!) because it generates all possibilities
        if best_sequence is None :
            best_sequence = sequence
        populate_costs(sequence,cost_array,instance) # O(n * m)
        last_cost = cost_array[jobs_count-1,machine_count-1]
        if last_cost < Cmax:
            best_sequence = sequence
            Cmax = last_cost
    return {
        "C_max" : Cmax,
        "order" : list(best_sequence)
    }

def get_results(instance : Instance):
    """Get the results of the algorithm on the specified instance

    Args:
        instance (class:instance): an FPS instance

    Returns:
        list: jobs order
    """
    return brute_force(instance)
