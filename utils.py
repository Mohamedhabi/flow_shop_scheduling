import numpy as np
from datetime import datetime
import re

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
            for i, group in enumerate(get_groups(file, "number of jobs")):
                self.number_of_instances = i+1
                group[1] = re.sub(' +', ' ', group[1])[1:-1]
                infos = group[1].split(' ')
                instance_matrix = []
                
                for i in range(3, 3+nb_machines):
                    group[i] = group[i][:-1]
                    machine = group[i].split(' ')
                    machine = [ x for x in machine if x.isdigit() ]
                    instance_matrix.append(machine)

                self.instances.append(Instance(
                    instance_2d_array = np.array(instance_matrix,dtype= np.int64).transpose(), 
                    upper_bound = int(infos[-2]), 
                    lower_bound = int(infos[-1])))

    def get_instances_number(self):
        return self.number_of_instances
    
    #index starts at 0
    def get_instance(self, number):
        if number < self.number_of_instances:
            return self.instances[number]