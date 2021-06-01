import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Instance, Benchmark
from fsp import ACO, specific_heuristic_NEH,branch_and_bound, meta_heuristic_ga, simulated_annealing, CDS, GUPTA
import numpy as np
import json
import re


OUTPUT_FOLDER = 'results'

def get_result_file_name(method, jobs_number, machines_number, instance):
    return OUTPUT_FOLDER+'/'+method+'/b_'+ '%d_%d' % (jobs_number,machines_number)+'/res_'+'%d' % (instance)+".json"

def get_normalize_json(results):
    for result in results:
        result['results']["order"] = '['+", ".join(str(e) for e in result['results']["order"])+']'
    
    text_json = json.dumps(results, indent = 2)
    tmp = re.sub(r'\]"', ']', text_json)
    return  re.sub(r'"\[', '[', tmp)

def transfor_params_json(params, tai_benchmarks):
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
                    bench["params"].append(param)
                    break

        param.pop('benchmarks', None)
    return result

def run_test(module, folder_name, params_path, benchmarks_path, tai_benchmarks, nb_instances_per_benchmark = None):
    with open(params_path) as f:
        params = json.load(f)

    benchmark_paeams = transfor_params_json(params, tai_benchmarks)
    
    for b in benchmark_paeams:
        benchmark = Benchmark(b['jobs'], b['machines'], benchmark_folder = benchmarks_path)
        if nb_instances_per_benchmark is None:
            instances_number = benchmark.get_instances_number()
        else: 
            instances_number = nb_instances_per_benchmark
        for nb in range(instances_number):
            results = []
            instance = benchmark.get_instance(nb)
            if instance is None:
                break
            i = 0
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
                print('done', b['jobs'], b['machines'], nb, i)
                i += 1

            with open(get_result_file_name(folder_name, b['jobs'], b['machines'], nb), 'w+') as f:
                json.dump(results , f, indent = 2)

if __name__ == '__main__':
    #Benchmarks to execute
    tai_benchmarks = [
        [5,5],
        [20,5],
        [20,10],
        [20,20],
        [50,5],
        [50,10],
        [50,20],
        [100,5],
        [100,10],
        [100,20],
        [500,20]
    ]
    #run_test(ACO, 'aco', 'tests/params/aco.json', './benchmarks', tai_benchmarks, None)
    #run_test(simulated_annealing, 'sa', 'tests/params/sa.json', './benchmarks', tai_benchmarks, 2)
    #run_test(CDS, 'cds', 'tests/params/none.json', './benchmarks', tai_benchmarks)
    run_test(GUPTA, 'gupta', 'tests/params/none.json', './benchmarks', tai_benchmarks)
    #run_test(specific_heuristic_NEH, 'neh', 'tests/params/none.json', './benchmarks', tai_benchmarks)
    #run_test(meta_heuristic_ga, 'ga', 'tests/params/ga.json', './benchmarks', tai_benchmarks, 2)