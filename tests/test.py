import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Instance, Benchmark
from fsp import ACO, specific_heuristic_NEH,branch_and_bound, meta_heuristic_ga
import numpy as np
import json
import re


OUTPUT_FOLDER = 'results'

tai_benchmarks = [
    (5,5),
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

def get_result_file_name(method, jobs_number, machines_number, instance):
    return OUTPUT_FOLDER+'/'+method+'/b_'+ '%d_%d' % (jobs_number,machines_number)+'/res_'+'%d' % (instance)+".json"

def get_normalize_json(results):
    for result in results:
        result['results']["order"] = '['+", ".join(str(e) for e in result['results']["order"])+']'
    
    text_json = json.dumps(results, indent = 2)
    tmp = re.sub(r'\]"', ']', text_json)
    return  re.sub(r'"\[', '[', tmp)

def transfor_params_json(params):
    result =[]
    for b in tai_benchmarks:
        result.append({
            "jobs": b[0],
            "machines": b[1],
            "params": []
        })

    for param in params:
        for b in param['benchmarks']:
            for bench in result:
                if bench["jobs"] == b[0] and bench["machines"] == b[1]:
                    element = bench
                    break
            element["params"].append(param)
        param.pop('benchmarks', None)
    return result

def run_test(module):
    with open('tests/params/aco.json') as f:
        params = json.load(f)

    benchmark_paeams = transfor_params_json(params)
    
    for b in benchmark_paeams:
        benchmark = Benchmark(b['jobs'], b['machines'], benchmark_folder = './benchmarks')
        instances_number = benchmark.get_instances_number()
        for nb in range(instances_number):
            results = []
            instance = benchmark.get_instance(nb)
            for param in b["params"]:
                param_results = {
                    'params': param,
                    'results': []
                }
                results.append(param_results)                    
                param_results['results'].append({
                    'instance': nb,
                    'results': module.get_results(instance, **param)
                })
                print('done', b['jobs'], b['machines'], nb)

            with open(get_result_file_name('aco', b['jobs'], b['machines'], nb), 'w+') as f:
                json.dump(results , f, indent = 2)

if __name__ == '__main__':
    run_test(ACO)