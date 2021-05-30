import sys
from flask import Flask, request, jsonify,redirect
from flask_cors import CORS,cross_origin
from fsp import branch_and_bound, parallel_bnb
from utils import Instance, Benchmark,JsonBenchmark, gen_instance_id
from core.executor import execute
import numpy as np
from multiprocessing import Process, Value
from time import sleep
import json
import os.path
import os

app = Flask(__name__)
cors = CORS(app)
OUTPUT_FOLDER = 'results'

def get_result_file_name(jobs_number,machines_number,instance_number):
    return OUTPUT_FOLDER+'/bnb/res_'+ '%d_%d_%d' % (jobs_number,machines_number,instance_number)+".json"

def instance_file_to_numbers(file):
    file_split = file.split('_')
    return {
        'jobs_number': int(file_split[1]),
        'machines_number': int(file_split[2]),
        'instance_number': int(file_split[3][0])
    } 

def benchmark_file_to_numbers(file):
    #file exemple tai5_5.txt we want to extract (5, 5)
    file = file[3:-4]
    file_split = file.split('_')
    return {
        'jobs_number': int(file_split[0]),
        'machines_number': int(file_split[1]),
    } 

def read_all_instances(BENCHMARKS_FOLDER):
    results = {}
    instance_id = ''
    for file in os.listdir(BENCHMARKS_FOLDER):
        if file.endswith(".txt"):
            result = benchmark_file_to_numbers(file)
            benchmark = Benchmark(result['jobs_number'], result['machines_number'], BENCHMARKS_FOLDER)
            instances_number = benchmark.get_instances_number()
            for i in range(instances_number):
                results[gen_instance_id(result['jobs_number'], result['machines_number'], i)] = benchmark.get_instance(i)
    return results

#key:id instance -> value: Instance object
instances = read_all_instances('./benchmarks')

# def run_bnb(jobs_number,machines_number,instance_number):
#     jsonbenchmark = JsonBenchmark(jobs_number,machines_number,benchmark_folder="./benchmarks")
#     instance = jsonbenchmark.get_instance_by_index(instance_number)["instance"]
#     instance = Instance(np.asarray(instance))
#     results = branch_and_bound.get_results(instance,search_strategy=branch_and_bound.DEPTH_FIRST_SEARCH)
#     print(results)
#     with open(get_result_file_name(jobs_number,machines_number,instance_number), 'w+') as f:
#         json.dump(results , f)
    
@app.route('/')
def server():
    return redirect("http://localhost:3000", code=302)

@app.route('/run',methods=["POST"])
def run_method():
    #TODO: implement run method
    print("run..")
    body = request.get_json(force=True)
    print(body)
    instance_id = body.get("instance_id",None)
    print(instance_id)
    if instance_id is not None:
        instance = instances[instance_id]
        if instance is None:
            return jsonify({"error" : True,"message" : "Instance not found"}),404
    else:
        instanceJson = body.get("instance",None)
        if instanceJson is None :
            return jsonify({"error" : True,"message" : "instance id or instance must be provided"}),404
        instance = Instance.create_instance_from_json(instanceJson)
    res,err = execute(
        body.get("method_id",None),
        instance,
        body.get("params",None)
    )
    if err is not None:
        jsonify({"error" : True,"message" : err["message"]}),500
    return jsonify(res),200  


#http://localhost:5000/lunchbnb?jobs=5&machines=4&instance=1
# @app.route("/lunchbnb")
# def bnb():
#     jobs_number = int(request.args.get('jobs'))
#     machines_number = int(request.args.get('machines'))
#     instance_number = int(request.args.get('instance'))
#     p = Process(target=run_bnb,  args=(jobs_number,machines_number,instance_number))
#     p.start()
#     return 'Done'
    
#http://localhost:5000/bnb?jobs=5&machines=4&instance=1
# @app.route("/bnb")
# def results_bnb():
#     jobs_number = int(request.args.get('jobs'))
#     machines_number = int(request.args.get('machines'))
#     instance_number = int(request.args.get('instance'))
#     file_name = get_result_file_name(jobs_number,machines_number,instance_number)
#     if os.path.isfile(file_name):
#         with open(file_name) as file:
#             return json.load(file) 
#     else:
#         return 'no results for this instance'

# @app.route("/bnballresults")
# def all_results_bnb():
#     results = []
#     for file in os.listdir(OUTPUT_FOLDER+'/bnb'):
#         if file.endswith(".json"):
#             results.append(instance_file_to_numbers(file))
#     return jsonify(results)

@app.route("/instances",methods=["GET"])
@cross_origin()
def get_instance():
    instance_id = request.args.get('instance_id')
    instance = instances[instance_id]
    if(instance is not None):
        return jsonify({
            "error" : False, 
            "jobs" : instance.get_machines_number(),
            "machines" : instance.get_jobs_number(),
            "id" : instance_id,
            "instance" :instance.np_array.tolist()
        })
    else:
        return jsonify({"error" : True,"message" : f"no existing instance for id={instance_id}"})

@app.route("/instances/all",methods=["GET"])
@cross_origin()
def get_all_instances():
    result = []
    for k,v in instances.items():
        result.append({
            "jobs" : v.get_jobs_number(),
            "machines" :v.get_machines_number(),
            "id" : v.id,
            "instance" :v.np_array.tolist()
        })

    print("\n\n\n\n\n\n",'hey')
    return jsonify({
            "error" : False, 
            "count" : len(result),
            "instances" :result
        })

if __name__ == '__main__':
    app.debug = True
    app.run()