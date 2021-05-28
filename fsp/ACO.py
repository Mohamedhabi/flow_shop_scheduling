import numpy as np
import sys
sys.path.append("../")
import time
import multiprocessing
import ctypes

class Ant:
    def __init__(self, instance):
        self.scheduledJobs = []
        self.instance = instance
        self.makespan = 0

    def run(self, pheromone, heuristic_info, alpha, beta, q0, local_search, local_search_proba, local_search_nb_permutation, results_parallel = None):
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
            
            self.scheduledJobs.append(int(j))
        self.makespanMatrix = self.instance.makespan(self.scheduledJobs, return_matrix = True)
        self.makespan = self.makespanMatrix[-1][-1]    

        if local_search:
            for job_position in range(nb_jobs):
                positions = np.random.choice(range(nb_jobs), local_search_nb_permutation)
                
                for position in positions:
                    rand = np.random.uniform()
                    if rand < local_search_proba and position != job_position:
                        self.change_position(job_position, position)
        
        if results_parallel is not None:
            results_parallel['makespan'] = self.makespan
            results_parallel['sequence'] = self.scheduledJobs

        
    def change_position(self, from_position, to_position):
        machine_count = self.instance.get_machines_number()
        makespanMatrix = self.makespanMatrix.copy()
        cp = self.scheduledJobs.copy()
        job = self.scheduledJobs[from_position]
        cp.remove(job)
        cp.insert(to_position, job)
        continu_from = min(from_position, to_position)

        if continu_from == 0:
            job = cp[continu_from]
            makespanMatrix[continu_from][0] = self.instance.get_cost(job, 0)
            for machine in range(1, machine_count):
                cost = self.instance.get_cost(job, machine)
                left = makespanMatrix[continu_from][machine-1] 
                makespanMatrix[continu_from][machine] = left + cost
            continu_from += 1

        job_index = continu_from
        for job in cp[continu_from:]:
            top = makespanMatrix[job_index-1][0]
            makespanMatrix[job_index][0] = top + self.instance.get_cost(job, 0)
            for machine in range(1, machine_count):
                cost = self.instance.get_cost(job,machine)
                top = makespanMatrix[job_index-1][machine]
                left = makespanMatrix[job_index][machine-1] 
                makespanMatrix[job_index][machine] = max(top,left) + cost
            job_index += 1

        if makespanMatrix[-1][-1] < self.makespan:
            self.makespan = makespanMatrix[-1][-1]
            self.makespanMatrix = makespanMatrix
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
    
    def update_pheromone(self, makespan, scheduledJobs):
        i = 0
        c_max = makespan
        added_quantity = self.rho * self.Z / c_max
        for j in scheduledJobs:
            self.pheromoneMatrix[i,j] = (1 - self.rho) * self.pheromoneMatrix[i,j] + added_quantity
            i = j + 1

    def lunch_ants(self, pheromoneMatrix,results, barrier ,barrier_after_update,ants, nb_rounds, local_search , local_search_proba, local_search_nb_permutation):
        for _ in range(nb_rounds):
            for ant in ants:
                ant.run(pheromoneMatrix, self.heuristic_info, self.alpha, self.beta, self.q0, local_search, local_search_proba, local_search_nb_permutation, results)
            
            barrier.wait()
            barrier_after_update.wait()
    
    def exec_parallel(self, threads, nb_rounds,local_search , local_search_proba, local_search_nb_permutation):
        jobs = []
        ants_nubmer = len(self.ants)
        if threads < ants_nubmer:
            chunk_number = -(-threads // threads)
        else:
            chunk_number = 1
            threads = ants_nubmer
        with multiprocessing.Manager() as manager:
            self.results = manager.list()
            for thread in range(threads):
                self.results.append(
                    manager.dict(
                        {
                        "makespan": 0,
                        "sequence": []
                    })
                )
            pheromoneMatrix_shape = self.pheromoneMatrix.shape
            shared_arr = multiprocessing.Array(ctypes.c_double, self.pheromoneMatrix.flatten())
            self.pheromoneMatrix = np.frombuffer(shared_arr.get_obj()).reshape(pheromoneMatrix_shape[0],-1)
            self.best_results = manager.dict({"makespan": 0, "sequence": []})
            barrier = multiprocessing.Barrier(threads, action=self.round_update_parallel)
            barrier_after_update = multiprocessing.Barrier(threads)
            for i in range(0, threads-1):
                thread = multiprocessing.Process(target=self.lunch_ants,
                    args =  (
                        self.pheromoneMatrix,
                        self.results[i],
                        barrier,
                        barrier_after_update,
                        self.ants[i * chunk_number:(i+1) * chunk_number],
                        nb_rounds,
                        local_search , 
                        local_search_proba,
                        local_search_nb_permutation))
                jobs.append(thread)
            
            # The final thread
            thread = multiprocessing.Process(target=self.lunch_ants,
                    args = (
                        self.pheromoneMatrix,
                        self.results[threads - 1],
                        barrier,
                        barrier_after_update,
                        self.ants[threads-1 * chunk_number:ants_nubmer],
                        nb_rounds,
                        local_search, 
                        local_search_proba,
                        local_search_nb_permutation))
            jobs.append(thread)

            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            self.makespan = self.best_results['makespan']
            self.best_sequence = self.best_results['sequence']
    
    def round_update(self):
        best_ant = max(self.ants, key=lambda x: x.makespan)
        self.update_pheromone(best_ant.makespan, best_ant.scheduledJobs)

        if best_ant.makespan < self.makespan or self.makespan == 0:
             self.makespan = best_ant.makespan
             self.best_sequence = best_ant.scheduledJobs
    
    def round_update_parallel(self):
        #print('hey')
        best_ant = min(self.results, key=lambda x: x['makespan'])
        self.update_pheromone(best_ant['makespan'], best_ant['sequence'])

        if best_ant['makespan'] < self.best_results['makespan']or self.best_results['makespan'] == 0:
            self.best_results['makespan'] = best_ant['makespan']
            self.best_results['sequence'] = best_ant['sequence']
    
    def run(self, nb_rounds = 10, parallel = False, threads = 8, local_search = False, local_search_proba = 0.02, local_search_nb_permutation = 3):
        start = time.perf_counter()
        if parallel:
            self.exec_parallel(threads, nb_rounds, local_search , local_search_proba, local_search_nb_permutation)       
        else:
            for _ in range(nb_rounds):
                for ant in self.ants:
                    ant.run(self.pheromoneMatrix, self.heuristic_info, self.alpha, self.beta, self.q0, local_search, local_search_proba, local_search_nb_permutation)
                self.round_update()

        return {
            "C_max" :  self.makespan,
            "order" : self.best_sequence,
            "time" : time.perf_counter() - start,
        }

def get_results(
    instance, initValue = 10**(-6), nbAnts = 5, rho = 0.01, alpha = 1, beta = 0.0001, q0 = 0.97, heuristic_info_strategy = 'min', 
    nb_rounds = 2500, parallel = False, threads = 12, local_search = True, local_search_proba = 0.02, local_search_nb_permutation = 3):
    
    colony = Colony(instance, initValue, nbAnts, rho, alpha, beta, q0, heuristic_info_strategy)
    return colony.run(nb_rounds, parallel, threads, local_search, local_search_proba, local_search_nb_permutation)