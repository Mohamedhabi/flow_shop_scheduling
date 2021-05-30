#import utils
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import time
import random
import sys
import numpy as np
from utils import Instance,Benchmark
# from fsp.branch_and_bound import evaluateSequence


def evaluateSeqeunce(instance: Instance,sequence : tuple):
    jobs_count = instance.get_jobs_number()
    machine_count = instance.get_machines_number()
    cost_array = np.zeros((jobs_count,machine_count))
    job_index = 0
    for job in sequence:
        for machine in range(machine_count):
            cost = instance.get_cost(job,machine)
            top = 0 if job_index == 0 else cost_array[job_index-1][machine]
            left = 0 if machine == 0 else cost_array[job_index][machine-1] 
            cost_array[job_index][machine] = max(top,left) + cost
        job_index += 1
    return cost_array[jobs_count-1][machine_count-1]

instance1 = Instance(
    np.array([
        [2, 1],
        [5, 7],
        [0, 1],
        [4, 4]
    ], dtype=np.int64)
)

instance3 = Instance(
    np.array([
        [3,2,3],
        [1,4,2],
        [3,2,1],
    ], dtype=np.int64)
)

instance4 = Instance(
    np.array([
        [1,2,3,2],
        [1,4,2,10],
        [3,2,1,5],
        [4,10,3,1],
        [1,5,4,4],
        [2,3,2,6],
        [5,2,1,1],
    ], dtype=np.int64)
)


class Solution(object):
    """Implements functions and data structures for the problem and solution.

    Hold the processing times for the problem as well as the solution sequence,
    makesppan and idle time. Also implement functions for calculating completion
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
        # Processing times - numpy 2d array
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
		e: 2d array with the completion time of each job in each machine or
		completion time of the last job in the last machine (int)
	"""

    sequence_np = np.array(sol.sequence)
    sequence_length = len(sequence_np)
    num_machines=sol.num_machines
    processing_times=sol.processing_times
    # for i in range(sol.num_jobs()): 
    #     processing_times.append(sol.get_job_costs(i))
    e = np.zeros((sequence_length+1,num_machines+1), dtype='int64')

    for i in range(1,sequence_length+1 ):
	    e[i,0]=0
	    for j in range(1,num_machines+1):
		    if i == 0:
			    e[0,j]=0

		    if e[i - 1, j] > e[i, j - 1]:
			    e[i, j] = e[i - 1][j] + processing_times[sequence_np[i - 1] - 1][j-1]
		    else:
			    e[i, j] = e[i][j - 1] + processing_times[sequence_np[i - 1]- 1][j-1]


	# Return  makespan (integer)
    #print("e",e)
    sol.makespan=e[sequence_length, num_machines]
    return e[sequence_length, num_machines]


def tie_breaking(processing_times,e,f,ms,inserting_job,best_position,sequence_length, num_machines): 
    print("in tie breaking")
    best_makespan=ms[best_position]
    itbp=sys.maxsize
    num_ties=0
    fl=np.zeros((sequence_length+2,num_machines+2), dtype='int64')

    for i in range(1, sequence_length+1): 
        if ms[i]==best_makespan: 
            it=0
            num_ties+=1
            if i == sequence_length: 
                for j in range(1, num_machines+1): 
                    it += f[sequence_length][j]-e[sequence_length-1][j]-processing_times[inserting_job-1][j-1]
            else: 
                fl[i][1]=f[i][1]+processing_times[i-1][0]
                for j in range(2, num_machines+1): 
                    it += f[sequence_length][j]-e[sequence_length-1][j]-processing_times[inserting_job-1][j-1]
                    if fl[i][j-1]-f[i][j]>0: 
                        it+=fl[i][j-1]-f[i][j]

                    if fl[i][j-1]>f[i][j]: 
                        fl[i][j]=fl[i][j-1]+processing_times[i-1][j-1]
                    else: 
                        fl[i][j]=fl[i][j]+processing_times[i-1][j-1]
            if it<itbp: 
                best_position=i
                itbp=it
    return best_position
                    



	
def taillard_acceleration(sequence,processing_times,inserting_job, num_machines,use_tie_breaking): 
    sequence_length = len(sequence)

    iq = sequence_length+1
    e = np.zeros((sequence_length+2,num_machines+2), dtype='int64')
    q = np.zeros((sequence_length+2,num_machines+2), dtype='int64')
    f = np.zeros((sequence_length+2,num_machines+2), dtype='int64')
    ms = np.zeros(sequence_length+2, dtype='int64')

    for i in range(1,sequence_length + 2):
        if i < sequence_length+1 :
            iq =iq-1

            e[i][0] = 0
            q[iq][num_machines+1] = 0

        f[i][0] = 0
        jq = num_machines + 1
        for j in range(1,num_machines+1):
            if i == 1: #first job
                e[0][j] = 0
                q[sequence_length+1][num_machines +1- j] = 0
            if i < sequence_length+1 :
                jq = jq - 1
                if e[i][j - 1] > e[i - 1][j]:
                    e[i][j] = e[i][j - 1] + processing_times[sequence[i - 1]-1][ j-1]
                else: 
                    e[i][j] = e[i - 1][j] + processing_times[sequence[i - 1] - 1][ j-1]
                    
                if q[iq][jq + 1] > q[iq + 1][jq]:
                    q[iq][jq] = q[iq][jq + 1] + processing_times[sequence[iq - 1]-1][ jq-1]
                else: 
                    q[iq][jq] = q[iq + 1][jq] + processing_times[sequence[iq - 1] - 1][ jq-1]
                
            if f[i][j - 1] > e[i - 1][j]:
                 f[i][j] = f[i][j - 1] + processing_times[inserting_job - 1][ j-1]
            else: 
                f[i][j] = e[i - 1][j] + processing_times[inserting_job - 1][ j-1]
    
    # print("f[]: ", f)
    # print("e[]: ", e)
    # print("q[]: ", q)
   

    best_makespan = 0
    best_position = 0
    for i in range(1,sequence_length + 2):
        ms[i] = 0
        for j in range(1,num_machines+1):
            tmp = f[i][j] +	q[i][j]
            if tmp > ms[i]:
                ms[i] = tmp
        if (ms[i] < best_makespan or best_makespan == 0):
            best_makespan = ms[i]
            best_position = i
    if (use_tie_breaking>1): 
        best_position=tie_breaking(processing_times,e,f,ms,inserting_job,best_position,sequence_length,num_machines)
    # print("ms[]: ", ms)
    # print("best_position: ", best_position)
    return best_position, best_makespan


def insert_best_position(solution, job,tie_breaking=False):
        """ Insert the given job in the position that minimize makespan.

        Insert the job in the position at self.sequence that minimizes the
        sequence makespan.

        Arguments
            job: Job to be inserted (int).
           
        Returns
            makespan: Makespan after inserting the job.
        """
        if tie_breaking:
            use_tie_breaking = 1
        else:
            use_tie_breaking = 0

        processing_times=solution.processing_times
        # for i in range(solution.get_jobs_number()): 
        #     processing_times.append(solution.get_job_costs(i))

        sequence_np = np.array(solution.sequence, dtype='int64')
        best_position, makespan = taillard_acceleration(sequence_np,processing_times,job,solution.num_machines, use_tie_breaking)

        solution.sequence.insert(best_position- 1, int(job))
        solution.makespan=makespan
        return makespan


def NEH(inst,tie_breaking=False,order_jobs="SD"):

    processing_times=[]
    t0 = time.perf_counter()

    for i in range(inst.get_jobs_number()): 
         processing_times.append(inst.get_job_costs(i))

    solution=Solution(processing_times)

    total_processing_times=np.zeros(inst.get_jobs_number())
    # Order jobs

    for i in range(inst.get_jobs_number() ):
        
        total_processing_times[i]=np.sum(inst.get_job_costs(i))
    
    # for i in range(solution.get_jobs_number() ):
        
    #     total_processing_times[i]=np.sum(solution.get_job_costs(i))

    if order_jobs == "SD":
         sorted_jobs=total_processing_times.argsort()
    else: 
        if order_jobs == "AV":
            """Order jobs by non-decreasing sum of the mean and deviation (Huang and Chen, 2008)."""
            average_plus_deviation = dict()
            for i in range(1, solution.num_jobs + 1):
                avg = np.mean(solution.processing_times[i-1])
                dev = np.std(solution.processing_times[i-1])
                average_plus_deviation[i] = avg + dev
            sorted_jobs=sorted(average_plus_deviation, key=average_plus_deviation.get, reverse=True)
        else: 
            if order_jobs == "RD":
                sorted_jobs=total_processing_times.argsort()
                np.random.shuffle(sorted_jobs)
               
            else: 
                if order_jobs == "D":
                    sorted_jobs=np.flip(total_processing_times.argsort())
    
    #sorted_jobs = [i for i in range(1, solution.get_jobs_number()+1 )]
    #sorted_jobs=np.flip(total_processing_times.argsort())
    #sorted_jobs=total_processing_times.argsort()
    #print("sorted: ",sorted_jobs)

    # Take the first two jobs and schedule them in order to minimize the partial makespan
    solution.sequence= [sorted_jobs[0],sorted_jobs[1]]
    makespan1 = calculate_makespan(solution)
    #print("makespan1",makespan1)
    solution.sequence = [sorted_jobs[1], sorted_jobs[0]]
    # print("seq1",solution.sequence)
    # print("makespan2",calculate_makespan(solution))
    if makespan1 < calculate_makespan(solution):
        solution.sequence = [sorted_jobs[0],sorted_jobs[1]]
        #print("seq2",solution.sequence)
        solution.makespan = makespan1
    # For i = 3 to n: Insert the i-th job at the place, among
    # the i possible ones, which minimize the partial makespan
    exec_time = time.perf_counter() - t0
    for job in sorted_jobs[2:]:
        insert_best_position(solution,job,tie_breaking)
    for i in range(len(solution.sequence)):
        solution.sequence[i] = int(solution.sequence[i])
    #TMP Solution !!!!!!
    return {
        "C_max" : int(inst.makespan(solution.sequence)),
        "order" : solution.sequence,
        "time" : exec_time
    }
    
    # print("neh makespan",solution.makespan)
    # print("instance cmax",inst.makespan(solution.makespan))
    # print("mine makespan",calculate_makespan(solution))
    # print("b&b cmax", evaluateSeqeunce(inst,tuple(solution.sequence)))
    
def get_results(instance,tie_breaking=False,jobs_order="SD"):
    """Get the results of the algorithm on the specified instance

    Args:
        instance (class:instance): an FPS instance

    Returns:
        list: jobs order
    """
    return NEH(instance, tie_breaking, jobs_order)
# bn =Benchmark(20,20,"../benchmarks")
# ins = bn.get_instance(0)
# print(get_results(ins))
