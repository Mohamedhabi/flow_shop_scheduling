import time
import random
import numpy as np


class Instance:
    """
    Instance of the problem
    """
    def __init__(self, instance_2d_array, upper_bound = None, lower_bound = None):
        self.np_array = instance_2d_array
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
  

    def get_cost(self, job, machine):
        return self.np_array[job][machine]

    def get_job_costs(self, job):
        return self.np_array[job]

    def get_machine_costs(self, machine):
        return self.np_array[:,machine]

    def get_jobs_number(self):
        return self.np_array.shape[0]
    
    def get_machines_number(self):
        return self.np_array.shape[1]





# instance1 = Instance(
#     np.array([
#         [2, 1],
#         [5, 7],
#         [0, 1],
#         [4, 4]
#     ], dtype=np.int64)
# )

# instance3 = Instance(
#     np.array([
#         [3,2,3],
#         [1,4,2],
#         [3,2,1],
#     ], dtype=np.int64)
# )

# instance4 = Instance(
#     np.array([
#         [1,2,3,2],
#         [1,4,2,10],
#         [3,2,1,5],
#         [4,10,3,1],
#         [1,5,4,4],
#         [2,3,2,6],
#         [5,2,1,1],
#     ], dtype=np.int64)
# )


class Solution(object):
    """Implements functions and data structures for the problem and solution.

    Hold the processing times for the problem as well as the solution sequence,
    makespan. Also implement functions for calculating completion
    times, makespan, best insert position and idle time. Calculations are done
    in the compiled Cython module. Makespan and idle time must be integers.

    Attributes:
        num_jobs: Number of jobs to be sequenced (int).
        num_machines: Number of machines in the problem (int).
        makespan: Current makespan of the sequence (int, default: 0).
        sequence: List with the current sequence of jobs
    """

    def __init__(self, instance_processing_times):
        # int variables
        self.num_jobs = len(instance_processing_times)
        self.num_machines = len(instance_processing_times[0])
        self.makespan = 0

        # Current solution sequence - list object
        self.sequence = list()

        self.processing_times = instance_processing_times
    


def calculate_makespan(sol): 

    """Calculate completion times for each job in each machine.

	Arguments:
		sequence: Numpy array with Current sequence
		processing_times: Numpy 2d array with processing times.
		num_machines: Number of machines in this problem.
		return_array: If 1 return array with each completition time,
		if 0 return just an integer with the completion time of the
		last job in the last machine.

	Returns:
		e: completion time of the last job in the last machine (int)
	"""

    sequence_np = np.array(sol)
    sequence_length = sol.get_jobs_number()
    num_machines=sol.get_machines_number()
    #Get processing times 
    processing_times=[]
    for i in range(sol.get_jobs_number()): 
        processing_times.append(sol.get_job_costs(i))
    e = np.zeros((sequence_length+1,num_machines+1), dtype='int64')

    #Calculates makespan (the traditional way), used for the first two jobs
    for i in range(1,sequence_length+1 ):
	    e[i,0]=0
	    for j in range(1,num_machines+1):
		    if i == 0:
			    e[0,j]=0

		    if e[i - 1, j] > e[i, j - 1]:
			    e[i, j] = e[i - 1][j] + processing_times[i - 1][j-1]
		    else:
			    e[i, j] = e[i][j - 1] + processing_times[i - 1][j-1]


	# Return  makespan (integer)
    return e[sequence_length, num_machines]


#Calculates makepan more efficiently and returns the best position at which to to insert the job
def taillard_acceleration(sequence,processing_times,inserting_job, num_machines): 

    sequence_length = len(sequence)


    #first job is job at position 1 not 0
    iq = sequence_length+1
    e = np.zeros((sequence_length+2,num_machines+2), dtype='int64')
    q = np.zeros((sequence_length+2,num_machines+2), dtype='int64')
    f = np.zeros((sequence_length+2,num_machines+2), dtype='int64')
    ms = np.zeros(sequence_length+2, dtype='int64')

    for i in range(1,sequence_length + 1):
        if i < sequence_length :
            iq =iq-1
            e[i][0] = 0
            q[iq][num_machines+1] = 0
        f[i][0] = 0
        jq = num_machines + 1
        for j in range(1,num_machines+1):
            if i == 1: #first job
                e[0][j] = 0
                q[sequence_length+1][num_machines +1- j] = 0
                if i < sequence_length :
                    jq = jq - 1
                    if e[i][j - 1] > e[i - 1][j]:
                        e[i][j] = e[i][j - 1] + processing_times[i - 1][ j-1]
                    else: 
                        e[i][j] = e[i - 1][j] + processing_times[i - 1][ j-1]
                    
                    if q[iq][jq + 1] > q[iq + 1][jq]:
                        q[iq][jq] = q[iq][jq + 1] + processing_times[iq-1][ jq-1]
                    else: 
                        q[iq][jq] = q[iq + 1][jq] + processing_times[iq - 1][ jq-1]
                
                if f[i][j - 1] > e[i - 1][j]:
                    f[i][j] = f[i][j - 1] + processing_times[inserting_job - 1][ j-1]
                else: 
                    f[i][j] = e[i - 1][j] + processing_times[inserting_job - 1][ j-1]

    best_makespan = 0
    best_position = 0
    for i in range(1,sequence_length + 1):
        ms[i] = 0
        for j in range(1,num_machines+1):
            tmp = f[i][j] +	q[i][j]
            if tmp > ms[i]:
                ms[i] = tmp
            if (ms[i] < best_makespan or best_makespan == 0):
                best_makespan = ms[i]
                best_position = i
    return best_position, best_makespan


def insert_best_position(solution, job):
        """ Insert the given job in the position that minimize makespan.
            Uses taillard accelartion

        Arguments
            solution: contains the current sequence
            job: Job to be inserted (int).
           
        Returns
            makespan: Makespan after inserting the job.
        """
        
        processing_times=[]
        for i in range(solution.get_jobs_number()): 
            processing_times.append(solution.get_job_costs(i))

        sequence_np = np.array(solution.sequence, dtype='int64')
        best_position, makespan = taillard_acceleration(sequence_np,processing_times,job,solution.get_machines_number())

        solution.sequence.insert(best_position - 1, job)
        return makespan


def NEH(solution):

    total_processing_times=np.zeros(solution.get_jobs_number())

    # Order jobs
    for i in range(solution.get_jobs_number() ):
        total_processing_times[i]=np.sum(solution.get_job_costs(i))
    sorted_jobs=np.flip(total_processing_times.argsort())

    # Take the first two jobs and schedule them in order to minimize the partial makespan
    solution.sequence= [sorted_jobs[0],sorted_jobs[1]]
    makespan1 = calculate_makespan(solution)
    solution.sequence = [sorted_jobs[1], sorted_jobs[0]]
    if makespan1 < calculate_makespan(solution):
        solution.sequence = [sorted_jobs[0],sorted_jobs[1]]
        makespan = makespan1
    # For i = 3 to n: Insert the i-th job at the place, among
    # the i possible ones, which minimize the partial makespan
    for job in sorted_jobs[2:]:
        makespan=insert_best_position(solution,job)
    print(solution.sequence)
    return {"sequence": solution.sequence, "makespan":makespan}



# benchmark=Benchmark(20,5,benchmark_folder='../benchmarks')
# ben=benchmark.get_instance(0)










def get_results(instance):
    """Get the results of the algorithm on the specified instance

    Args:
        instance (class:instance): an FPS instance

    Returns:
        list: jobs order
    """
    start=time.perf_counter()
    results=NEH(instance)
    print(results)
    end=time.perf_counter()
    perf=print(end-start)
    print(perf)
    return {
        "C_max": results.makespan,
        "order":results.sequence,
        "time":perf
        }

