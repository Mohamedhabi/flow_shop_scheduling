import numpy as np
from datetime import datetime

def convert_to_datetime(x):
      return datetime.fromtimestamp(31536000+x*24*3600).strftime("%Y-%m-%d")

class Instance:
    """
    We dont know how the input is gonna be, so this classe would be an abstraction
    we'll update it when we have the input format
    """
    def __init__(self, instance_2d_array):
        self.np_array = instance_2d_array

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

