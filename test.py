from utils import Instance, Benchmark
from fsp import ACO, specific_heuristic_NEH,branch_and_bound
import numpy as np
import json

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

def test_ACO():
    for b in tai_benchmarks:
        benchmark = Benchmark(b[0], b[1], benchmark_folder = './benchmarks')
        instances_number = benchmark.get_instances_number()
        results = []
        for nb in range(instances_number):
            instance = benchmark.get_instance(nb)
            results.append({
                'instance': nb,
                'results': ACO.get_results(instance)
            })
        with open(get_result_file_name('aco', b[0], b[1]), 'w+') as f:
            json.dump(results , f)

test_ACO()