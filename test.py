from numpy.core.numeric import _rollaxis_dispatcher
from utils import Instance, Benchmark
from fsp import ACO, specific_heuristic_NEH,branch_and_bound, meta_heuristic_ga
import numpy as np
import json
import re
from fsp.branch_and_bound import evaluateSeqeunce
OUTPUT_FOLDER = 'results'

tai_benchmarks = [
    (20,5),
    (20,10),
    (20,20),
    (50,5),
    (50,10),
    (50,20),
    (100,5),
    (100,10),
    (100,20),
    (500,20),
    ]

def get_result_file_name(method,jobs_number,machines_number):
    return OUTPUT_FOLDER+'/'+method+'/res_'+ '%d_%d' % (jobs_number,machines_number)+".json"

def get_normalize_json(results):
    for result in results:
        result['results']["order"] = '['+", ".join(str(e) for e in result['results']["order"])+']'
    
    text_json = json.dumps(results, indent = 2)
    tmp = re.sub(r'\]"', ']', text_json)
    return  re.sub(r'"\[', '[', tmp)

def test_ACO():
    for b in tai_benchmarks:
        benchmark = Benchmark(b[0], b[1], benchmark_folder = './benchmarks')
        instances_number = benchmark.get_instances_number()
        results = []
        for nb in range(1):
            instance = benchmark.get_instance(nb)
            results.append({
                'instance': nb,
                'results': ACO.get_results(instance)
            })
        with open(get_result_file_name('aco', b[0], b[1]), 'w+') as f:
            json.dump(results , f, indent = 2)

def test_NEH():
    for b in tai_benchmarks:
        benchmark = Benchmark(b[0], b[1], benchmark_folder = './benchmarks')
        instances_number = benchmark.get_instances_number()
        results = []
        for nb in range(1):
            instance = benchmark.get_instance(nb)
            results.append({
                'instance': nb,
                'results': specific_heuristic_NEH.get_results(instance)
            })
        with open(get_result_file_name('neh', b[0], b[1]), 'w+') as f:
            json.dump(results , f, indent = 2)

def test_ga():
    for b in tai_benchmarks:
        benchmark = Benchmark(b[0], b[1], benchmark_folder = './benchmarks')
        instances_number = benchmark.get_instances_number()
        results = []
        for nb in range(instances_number):
            instance = benchmark.get_instance(nb)
            results.append({
                'instance': nb,
                'results': meta_heuristic_ga.get_results(instance)
            })

        with open(get_result_file_name('ga', b[0], b[1]), 'w+') as f:
            f.write(get_normalize_json(results))

#test_ga()
benchmark = Benchmark(20,20, benchmark_folder = './benchmarks')
instance = benchmark.get_instance(0)
print([13, 2, 19, 8, 4, 11, 9, 10, 1, 5, 17, 14, 6, 12, 7, 15, 3, 16, 0, 18])
print(instance.makespan([13, 2, 19, 8, 4, 11, 9, 10, 1, 5, 17, 14, 6, 12, 7, 15, 3, 16, 0, 18]))
print(evaluateSeqeunce(instance,tuple([13, 2, 19, 8, 4, 11, 9, 10, 1, 5, 17, 14, 6, 12, 7, 15, 3, 16, 0, 18])))