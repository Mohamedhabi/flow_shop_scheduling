import numpy as np
from datetime import datetime
import json
import re

from sympy import Id

def convert_to_datetime(x):
      return datetime.fromtimestamp(31536000+x*24*3600).strftime("%Y-%m-%d")

def get_groups(seq, group_by):
    data = []
    for line in seq:
        if line.startswith(group_by):
            if data:
                yield data
                data = []
        data.append(line)

    if data:
        yield data

class Instance:
    """
    Instance of the problem
    """
    def __init__(self, instance_2d_array,id=None, upper_bound = None, lower_bound = None):
        self.id = id
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

    def get_array(self):
        return self.np_array

    def get_chart_data(self, results):
        # result: {'C_max': 13, 'order': [0, 1, 2]}
        machine_free = np.zeros(self.get_machines_number())
        df = []
        current_time = 0

        for job in results['order']:
            current_time = 0
            for machine in range(self.get_machines_number()):
                start = max(current_time, machine_free[machine])
                end = max(current_time, machine_free[machine]) + self.get_cost(job, machine)
                machine_free[machine] = end
                current_time = end
                df.append(dict(Task="Machine "+str(machine), Start=convert_to_datetime(int(start)), Finish=convert_to_datetime(int(end)), Resource="Job "+str(job)))
        
        num_tick_labels = np.linspace(start = 0, stop = int(current_time), num = int(current_time+1), dtype = int)
        date_ticks = [convert_to_datetime(x) for x in num_tick_labels]

        return {
            'df':df,
            'num_tick_labels':num_tick_labels,
            'date_ticks':date_ticks
            }

    def makespan(self, schedule):
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
        jobs_count = self.get_jobs_number()
        machine_count = self.get_machines_number()
        cost_array = np.zeros((jobs_count,machine_count))
        job_index = 0
        for job in schedule:
            for machine in range(machine_count):
                cost = self.get_cost(job,machine)
                top = 0 if job_index == 0 else cost_array[job_index-1][machine]
                left = 0 if machine == 0 else cost_array[job_index][machine-1] 
                cost_array[job_index][machine] = max(top,left) + cost
            job_index += 1
        return cost_array[jobs_count-1][machine_count-1]

def gen_instance_id(jobs: int,machine: int,index: int):
    return str(jobs)+"x"+str(machine)+"-"+str(index)
class Benchmark:
    """
    A class representing a benchmark
    a benchmark is consists of multiple instance 
    """
    def __init__(self, nb_jobs, nb_machines, benchmark_folder = './benchmarks'):
        self.nb_jobs = nb_jobs
        self.nb_machines = nb_machines
        self.instances = []
        self.number_of_instances = 0
        with open(benchmark_folder+"/tai"+str(nb_jobs)+'_'+str(nb_machines)+".txt") as file:
            for j, group in enumerate(get_groups(file, "number of jobs")):
                self.number_of_instances = j+1
                group[1] = re.sub(' +', ' ', group[1])[1:-1]
                infos = group[1].split(' ')
                instance_matrix = []
                
                for i in range(3, 3+nb_machines):
                    group[i] = group[i][:-1]
                    machine = group[i].split(' ')
                    machine = [ x for x in machine if x.isdigit() ]
                    instance_matrix.append(machine)

                self.instances.append(Instance(
                    instance_2d_array = np.array(instance_matrix,dtype= np.int64).transpose(),id=gen_instance_id(nb_jobs,nb_machines,j),
                    upper_bound = int(infos[-2]), 
                    lower_bound = int(infos[-1])))

    def get_instances_number(self):
        return self.number_of_instances
    
    #index starts at 0
    def get_instance(self, number):
        if number < self.number_of_instances:
            return self.instances[number]

class JsonBenchmark():
    def __init__(self, nb_jobs, nb_machines, benchmark_folder = './benchmarks'):
        self.nb_jobs = nb_jobs
        self.nb_machines = nb_machines
        self.instances = []
        self.number_of_instances = 0
        with open(benchmark_folder+"/instances_"+str(nb_jobs)+'_'+str(nb_machines)+".json") as file:
            data = json.load(file)
            self.instances = data
            self.number_of_instances = len(self.instances)

    def get_instances_number(self):
        return self.number_of_instances

    #index starts at 0
    def get_instance_by_index(self, index):
        if index < self.number_of_instances:
            return self.instances[index]   


# benchmark = JsonBenchmark(11,6)
# print(benchmark.get_instance_by_index(0)["instance"])