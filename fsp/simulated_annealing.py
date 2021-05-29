from random import randint
import time
import numpy as np
from .specific_heuristic_NEH import NEH
from .CDS import cds as CDS
from utils import Instance

#population_size = 100
#job_count = 20
#nb_machines = len(matrice[0])


def simulated_annealing(instance : Instance, intial_value , Ti = 2000,Tf = 0.1 ,nb_repetitions=100, alpha = 0.93):
    #Number of jobs given
    nb_machines = instance.get_machines_number()
    job_count = instance.get_jobs_number()
    n = job_count;


    #Initialize the primary seq
    if(intial_value=='NEH') :
        print('NEH')
        neh = NEH(instance)
        old_seq = neh['sequence']
        old_makeSpan = neh['makespan']  
        
    elif(intial_value=='CDS') :
        print('CDS')
        cds = CDS(instance)
        old_seq = cds['order']
        old_makeSpan = cds['C_max']  
    new_seq =[]  
    delta_mk1 = 0
    #Initialize the temperature
    T = Ti
    Tf = Tf
    alpha = alpha
    # of iterations
    temp_cycle = 0
    while (T >= Tf)  :
        for i in range(0,nb_repetitions-1) : 
            new_seq = old_seq.copy()
            job = new_seq.pop(randint(0,n-1))
            new_seq.insert(randint(0,n-1),job)        
            new_make_span = makespan(new_seq, instance)
            delta_mk1 = new_make_span - old_makeSpan
            if delta_mk1 <= 0:
                old_seq = new_seq
                old_makeSpan = new_make_span
            else :
                Aprob = np.exp(-(delta_mk1/T))
                if Aprob > np.random.uniform(0.5,0.9):
                    old_seq = new_seq
                    old_makeSpan = new_make_span
                else :
                    #The solution is discarded (on élague)
                    pass
        T = T * alpha 
        temp_cycle += 1


    #Result Sequence
    seq = old_seq
    schedules = np.zeros((nb_machines, job_count), dtype=dict)
    # schedule first job alone first
    task = {"name": "job_{}".format(
        seq[0] ), "start_time": 0, "end_time": instance.np_array[seq[0]][0]}
    schedules[0][0] = task

    for m_id in range(1, nb_machines):
        start_t = schedules[m_id - 1][0]["end_time"]
        end_t = start_t + instance.np_array[0][m_id]
        task = {"name": "job_{}".format(
            seq[0] ), "machine_id" : m_id ,"start_time": start_t, "end_time": end_t}
        schedules[m_id][0] = task

    for index, job_id in enumerate(seq[1::]):
        start_t = schedules[0][index]["end_time"]
        end_t = start_t + instance.np_array[job_id][0]
        task = {"name": "job_{}".format(
            job_id ), "start_time": start_t, "end_time": end_t}
        schedules[0][index + 1] = task
        for m_id in range(1, nb_machines):
            start_t = max(schedules[m_id][index]["end_time"],
                            schedules[m_id - 1][index + 1]["end_time"])
            end_t = start_t +instance.np_array[job_id][m_id]
            task = {"name": "job_{}".format(
                job_id ), "machine_id" : m_id , "start_time": start_t, "end_time": end_t}
            schedules[m_id][index + 1] = task
    return {
        "order" : seq, 
        "C_max" : schedules[-1][-1]['end_time'], 
        }

def makespan (jobOrder : list, jobMatrix : Instance) :
    nb_machines = int(jobMatrix.get_machines_number())
    nb_jobs = len(jobOrder)
    tab = np.zeros((nb_jobs, nb_machines*2))
    for i in range(0, nb_jobs):
        for j in range(0, nb_machines):
            tab[i, j*2] = max (tab[i-1,2*j+1], tab[i,2*j-1])
            tab[i, j*2+1] = tab[i, 2*j] + int(jobMatrix.np_array[jobOrder[i], j])
    return int(tab[-1,-1])

def get_results (instance : Instance, initial_value = 'CDS', Ti=2000, Tf = 0.1 ,nb_repetitions=100, alpha = 0.93):
    start = time.perf_counter()
    result = simulated_annealing(instance,initial_value, Ti, Tf,nb_repetitions,alpha)
    end = time.perf_counter()
    return {
        "order" : result['order'],
        'C_max' : result['C_max'],
        'time' : end-start
    }