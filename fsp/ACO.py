import numpy as np
import sys
sys.path.append("../")
from utils import Instance, Benchmark,JsonBenchmark
import threading
import time

class Ant:
    def __init__(self, instance):
        self.scheduledJobs = []
        self.instance = instance
        self.makespan = 0

    def run(self, pheromone, heuristic_info, alpha, beta, local_search = False, local_search_proba = 0.02):
        j = 0
        nb_jobs = self.instance.get_jobs_number()
        for _ in range(nb_jobs):
            unscheduledJobs_function = (pheromone[j]**alpha * heuristic_info** beta)
            # to eleminate already scheduled jobs
            unscheduledJobs_function[self.scheduledJobs] = -1
            j = unscheduledJobs_function.argmax()
            self.scheduledJobs.append(j)
        self.makespan = self.instance.makespan(self.scheduledJobs)

        if local_search:
            for job in range(nb_jobs):
                rand = np.random.uniform()
                if rand < local_search_proba:
                    for position in range(nb_jobs):
                        if self.scheduledJobs[position] != job:
                            cp = self.scheduledJobs.copy()
                            cp.remove(job)
                            cp.insert(position, job)
                            cp_makespan = self.instance.makespan(cp)
                            if (cp_makespan < self.makespan):
                                self.makespan = cp_makespan
                                self.scheduledJobs = cp


class Colony:
    def __init__(self, instance, initValue = 2, nbAnts = 5, rho = .5, alpha = 1, beta = 1, Z = 1,heuristic_info_strategy = 'min'):
        jobNum = instance.get_jobs_number()
        self.instance = instance
        # +1 to coonsider the initial state
        self.pheromoneMatrix = np.full((jobNum + 1, jobNum), initValue, dtype= np.float)
        self.ants = [Ant(instance) for i in range(nbAnts)]
        #evaporation rate
        self.rho = rho
        self.R1 = self.ExtensionOfJohnson(instance)
        self.R2 = self.ExtensionOfSPT(instance)
        self.alpha = alpha
        self.beta = beta
        self.Z = Z
        self.makespan = 0
        self.best_sequence = []
        
        if heuristic_info_strategy == 'max':
            self.heuristic_info = np.max((self.R1, self.R2), axis = 0)
        elif heuristic_info_strategy == 'mean':
            self.heuristic_info = np.mean((self.R1, self.R2), axis = 0)
        else:
            self.heuristic_info = np.min((self.R1, self.R2), axis = 0)

    def ExtensionOfJohnson(self, instance):
        instance_matrix = instance.get_array()
        two_first = instance_matrix[:,:2].sum(axis = 1)
        two_last = instance_matrix[:,-2:].sum(axis = 1)
        diff = two_last / two_first
        return diff / diff.sum()

    def ExtensionOfSPT(self, instance):
        inverseInstance = 1 / instance.get_array()
        sumJobs = inverseInstance.sum(axis = 1)
        return sumJobs / sumJobs.sum()
    
    def update_pheromone(self, best_ant):
        i = 0
        c_max = best_ant.makespan
        for j in best_ant.scheduledJobs:
            self.pheromoneMatrix[i,j] = (1 - self.rho) * self.pheromoneMatrix[i,j] + self.rho * self.Z / c_max
            i = j + 1

    def lunch_ants(self, start, end, local_search , local_search_proba):
        for ant in range(start,end):
            self.ants[ant].run(self.pheromoneMatrix, self.heuristic_info, self.alpha, self.beta, local_search, local_search_proba)


    def lunch_round(self, parallel = False, threads = 8, local_search = False, local_search_proba = 0.02):
        if parallel:
            jobs = []
            ants_nubmer = len(self.ants)
            if threads < ants_nubmer:
                chunk_number = -(-threads // threads)
                for i in range(0, threads-1):
                    thread = threading.Thread(target=self.lunch_ants(
                        i * chunk_number, 
                        (i+1) * chunk_number,
                        local_search , 
                        local_search_proba))
                    jobs.append(thread)
                
                # The final thread
                thread = threading.Thread(target=self.lunch_ants(
                        threads-1 * chunk_number, 
                        ants_nubmer,
                        local_search, 
                        local_search_proba))
                jobs.append(thread)

            else:
                for i in range(0, ants_nubmer):
                    thread = threading.Thread(target=self.lunch_ants(
                        i, 
                        i+1,
                        local_search, 
                        local_search_proba))
                    jobs.append(thread)

            for j in jobs:
                j.start()

            for j in jobs:
                j.join()
        else:
            for ant in self.ants:
                ant.run(self.pheromoneMatrix, self.heuristic_info, self.alpha, self.beta, local_search = True, local_search_proba = 0.02)

        best_ant = max(self.ants, key=lambda x: x.makespan)
        self.update_pheromone(best_ant)

        if best_ant.makespan < self.makespan or self.makespan == 0:
             self.makespan = best_ant.makespan
             self.best_sequence = best_ant.scheduledJobs
    
    def run(self, nb_rounds = 10, parallel = False, threads = 8, local_search = False, local_search_proba = 0.02):
        start = time.time()
        for _ in range(nb_rounds):
            self.lunch_round(parallel, threads, local_search, local_search_proba)
        return {
        "C_max" :  self.makespan,
        "order" : self.best_sequence,
        "time" : time.time() - start,
        }

def get_results(
    instance, initValue = 2, nbAnts = 5, rho = .5, alpha = 1, beta = 1, Z = 1, heuristic_info_strategy = 'min', 
    nb_rounds = 10, parallel = False, threads = 8, local_search = False, local_search_proba = 0.02):

    colony = Colony(instance, initValue, nbAnts, rho, alpha, beta, Z, heuristic_info_strategy)
    return colony.run(nb_rounds, parallel, threads, local_search, local_search_proba)

instance = Instance(
    np.array([
        [1,2,3,2],
        [1,4,2,10],
        [3,2,1,5],
        [4,10,3,1],
        [1,5,4,4],
        [2,3,2,6],
        [5,2,1,1],
        [2,3,2,6],
        [5,2,1,1],
    ], dtype=np.int64)
)

cl = Colony(instance)
# print(cl.R1)
# print(cl.R2)
# print(cl.heuristic_info)
import time
t0 = time.time()
cl.run(nb_rounds = 50, parallel = True, threads = 12, local_search = False, local_search_proba = 0.02)
print(time.time()-t0)
t0 = time.time()
cl.run(nb_rounds = 50, parallel = False, threads = 12, local_search = False, local_search_proba = 0.02)
print(time.time()-t0)
print(cl.makespan)