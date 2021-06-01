
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import numpy as np
import time

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
def get_results(instance: Instance):
    start = time.perf_counter() 
    partition = gupta_partition(instance); 
    U = partition['U']; 
    V = partition['V']; 
    U.sort(key=lambda x: x[1], reverse=True); 
    V.sort(key=lambda x: x[1], reverse=False); 
    U.extend(V); 
    schedule = []
    for i in U:
        schedule.append(i[0]); 
    C_max = instance.makespan(schedule); 
    return {
        "C_max": C_max,
        "order": schedule,
        "time": time.perf_counter() - start,
    } 