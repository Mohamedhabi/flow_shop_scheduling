import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy.core.numeric import _rollaxis_dispatcher
from utils import Instance, Benchmark
from fsp import ACO, specific_heuristic_NEH,branch_and_bound, meta_heuristic_ga
import numpy as np
import json
import re


OUTPUT_FOLDER = 'results'

tai_benchmarks = [
    (20,20),

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
    with open('tests/params/aco.json') as f:
        data = json.load(f)
        print(type(data[0]))
    for params in data:
        for b in tai_benchmarks:
            benchmark = Benchmark(b[0], b[1], benchmark_folder = './benchmarks')
            instances_number = benchmark.get_instances_number()
            results = {
                'params': params,
                'results': []
            }
            for nb in range(1):
                instance = benchmark.get_instance(nb)
                results['results'].append({
                    'instance': nb,
                    'results': ACO.get_results(instance, **params)
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

if __name__ == '__main__':
#test_ga()
    test_ACO()
    benchmark = Benchmark(20,20, benchmark_folder = './benchmarks')
    instance = benchmark.get_instance(0)
    print([16, 0, 1, 18, 14, 12, 3, 8, 6, 4, 13, 2, 19, 15, 11, 17, 10, 7, 9, 5])
    print(instance.makespan([16, 0, 1, 18, 14, 12, 3, 8, 6, 4, 13, 2, 19, 15, 11, 17, 10, 7, 9, 5]))