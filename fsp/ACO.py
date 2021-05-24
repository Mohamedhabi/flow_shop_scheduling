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

    def run(self, pheromone, heuristic_info, alpha, beta, q0, local_search = False, local_search_proba = 0.02):
        self.scheduledJobs = []
        j = -1
        nb_jobs = self.instance.get_jobs_number()
        for _ in range(nb_jobs):
            unscheduledJobs_function = (pheromone[j+1]**alpha * heuristic_info**beta)
            rand = np.random.uniform()
            if rand < q0:
                # to eleminate already scheduled jobs
                unscheduledJobs_function[self.scheduledJobs] = -1
                j = unscheduledJobs_function.argmax()
            else:
                # to give scheduled jobs a 0 probability
                unscheduledJobs_function[self.scheduledJobs] = 0
                sum = unscheduledJobs_function.sum()
                j = np.random.choice(list(range(nb_jobs)), 1, p = (unscheduledJobs_function / sum))[0]
            
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
    def __init__(self, instance, initValue = 2, nbAnts = 5, rho = .5, alpha = 1, beta = 1, q0 = 0.97,heuristic_info_strategy = 'min'):
        jobNum = instance.get_jobs_number()
        self.instance = instance
        # +1 to coonsider the initial state
        self.pheromoneMatrix = np.full((jobNum + 1, jobNum), initValue, dtype= np.float64)
        self.ants = [Ant(instance) for i in range(nbAnts)]
        #evaporation rate
        self.rho = rho
        self.R1 = self.ExtensionOfJohnson(instance)
        self.R2 = self.ExtensionOfSPT(instance)
        self.alpha = alpha
        self.beta = beta
        self.Z = instance.get_jobs_number() * instance.get_machines_number()
        self.makespan = 0
        self.best_sequence = []
        self.q0 = q0
        
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

    def lunch_ants(self, barrier, barrier_after_update,start, end, nb_rounds, local_search , local_search_proba):
        for _ in range(nb_rounds):
            for ant in range(start,end):
                self.ants[ant].run(self.pheromoneMatrix, self.heuristic_info, self.alpha, self.beta, self.q0, local_search, local_search_proba)
            
            id = barrier.wait()
            if id == 0:
                barrier.reset()
            id = barrier_after_update.wait()
            if id == 0:
                barrier_after_update.reset()
    
    def create_threads(self, threads, nb_rounds,local_search , local_search_proba):
        jobs = []
        ants_nubmer = len(self.ants)
        if threads < ants_nubmer:
            barrier = threading.Barrier(threads, action=self.round_update)
            barrier_after_update = threading.Barrier(threads)
            chunk_number = -(-threads // threads)
            for i in range(0, threads-1):
                thread = threading.Thread(target=self.lunch_ants(
                    barrier,
                    barrier_after_update,
                    i * chunk_number, 
                    (i+1) * chunk_number,
                    nb_rounds,
                    local_search , 
                    local_search_proba))
                jobs.append(thread)
            
            # The final thread
            thread = threading.Thread(target=self.lunch_ant,
                    args = (
                    barrier,
                    threads-1 * chunk_number, 
                    ants_nubmer,
                    nb_rounds,
                    local_search, 
                    local_search_proba))
            jobs.append(thread)
        else:
            barrier = threading.Barrier(ants_nubmer, action=self.round_update)
            barrier_after_update = threading.Barrier(ants_nubmer)
            for i in range(0, ants_nubmer):
                thread = threading.Thread(target=self.lunch_ants,
                    args =(
                        barrier,
                        barrier_after_update,
                        i, 
                        i+1,
                        nb_rounds,
                        local_search, 
                        local_search_proba))
                jobs.append(thread)
        return jobs
    
    def round_update(self):
        best_ant = max(self.ants, key=lambda x: x.makespan)
        self.update_pheromone(best_ant)

        if best_ant.makespan < self.makespan or self.makespan == 0:
             self.makespan = best_ant.makespan
             self.best_sequence = best_ant.scheduledJobs
    
    def run(self, nb_rounds = 10, parallel = False, threads = 8, local_search = False, local_search_proba = 0.02):
        start = time.time()
        if parallel:
            jobs = self.create_threads(threads, nb_rounds, local_search , local_search_proba)
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()        
        else:
            for _ in range(nb_rounds):
                for ant in self.ants:
                    ant.run(self.pheromoneMatrix, self.heuristic_info, self.alpha, self.beta, self.q0, local_search, local_search_proba)
                self.round_update()

        return {
        "C_max" :  self.makespan,
        "order" : self.best_sequence,
        "time" : time.time() - start,
        }

def get_results(
    instance, initValue = 10**(-6), nbAnts = 12, rho = 0.01, alpha = 1, beta = 0.0001, q0 = 0.97, heuristic_info_strategy = 'min', 
    nb_rounds = 2500, parallel = True, threads = 12, local_search = True, local_search_proba = 0.02):

    colony = Colony(instance, initValue, nbAnts, rho, alpha, beta, heuristic_info_strategy)
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
benchmark = Benchmark(200, 20, benchmark_folder = '../benchmarks')
instance = benchmark.get_instance(0)

cl = Colony(instance, nbAnts = 12)

print(cl.run(nb_rounds = 200, parallel = True, threads = 12, local_search = False, local_search_proba = 0.02))
cl = Colony(instance, nbAnts = 12)

print(cl.run(nb_rounds = 200, parallel = False, threads = 12, local_search = False, local_search_proba = 0.02))
print(instance.makespan([1, 0, 2, 4, 5, 7, 3, 8, 6]))
print(instance.makespan([0, 2, 1, 4, 5, 8, 3, 7, 6]))