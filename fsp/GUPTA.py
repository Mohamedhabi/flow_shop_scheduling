
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import numpy as np
from CDS import *
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
def gupta_indice(j,jobCosts):
    """an algorithem that calculates the indice of gupta for the job j with operations' time 

    Args:
        j : number of job 
        jobCosts: an array that contains all operations' times for job j 

    Returns:
            indice 
    """
    indice = jobCosts[0] + jobCosts[1]; 
    for i in range(len(jobCosts)-1):
        if(indice > jobCosts[i+1] + jobCosts[i]):  
            indice = jobCosts[i+1] + jobCosts[i]; 
    return indice; 

def gupta_partition(instance: Instance):
    """an algorithem that create two lists of jobs U and V from the instance jobs 

    Args:
        instance (class:instancdce): an FPS instance

    Returns:
        {
            "U" : a list of jobs that verify Tj1 < Tjm 
            "V" : J - U
        }
    """
    m = instance.get_machines_number(); 
    n = instance.get_jobs_number(); 
    U = []
    V = []
    for i in range(n):
        jobCosts = instance.get_job_costs(i); 
        if (jobCosts[0]<jobCosts[m-1]):
            indice = gupta_indice(i,jobCosts); 
            U.append((i,indice)); 
        else: 
            indice = gupta_indice(i,jobCosts); 
            V.append((i,indice)); 
    return {
        "U" : U,
        "V" : V
    }
def gupta(instance: Instance):
    partition = gupta_partition(instance); 
    U = partition['U']; 
    V = partition['V']; 
    U.sort(key=lambda x: x[1], reverse=True); 
    V.sort(key=lambda x: x[1], reverse=False); 
    U.extend(V); 
    schedule = []
    for i in U:
        schedule.append(i[0]); 
    C_max = makespan(instance,schedule); 
    return {
        "C_max": C_max,
        "order": schedule
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

benchmark = Benchmark(20, 5, benchmark_folder = '../benchmarks')
instance = benchmark.get_instance(2)
out=gupta(instance)
out1=cds(instance)
print(out['C_max'])
print(*out['order'])
print(out1['C_max'])
print(*out1['order'])