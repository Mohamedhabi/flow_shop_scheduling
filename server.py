import sys
from flask import Flask, request, jsonify,redirect
from flask_cors import CORS,cross_origin
from fsp import branch_and_bound, parallel_bnb
from utils import Instance, Benchmark,JsonBenchmark
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
benchmark20_20 = Benchmark(20,20)
def get_result_file_name(jobs_number,machines_number,instance_number):
    return OUTPUT_FOLDER+'/bnb/res_'+ '%d_%d_%d' % (jobs_number,machines_number,instance_number)+".json"

def instance_file_to_numbers(file):
    file_split = file.split('_')
    return {
        'jobs_number': int(file_split[1]),
        'machines_number': int(file_split[2]),
        'instance_number': int(file_split[3][0])
    } 

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
        instance = benchmark20_20.get_instance(instance_id)
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
        jsonify({"error" : True,"message" : err["message"]})
    return jsonify(res)   


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

instances = {
    20 : {
        20 : benchmark20_20
    }
}
@app.route("/instances",methods=["GET"])
@cross_origin()
def get_instance():
    print("hello")
    jobs_number = int(request.args.get('jobs'))
    machines_number = int(request.args.get('machines'))
    instance_number = int(request.args.get('instance'))
    try:
        benchmark = instances[jobs_number][machines_number]
    except :
        benchmark = None
    if(benchmark is not None):
        instance = benchmark.get_instance(instance_number)
        return jsonify({
            "error" : False, 
            "jobs" : jobs_number,
            "machines" :machines_number,
            "id" : instance.id,
            "instance" :instance.np_array.tolist()
        })
    else:
        return jsonify({"error" : True,"message" : f"no existing benchmark for jobs={jobs_number} and machines={machines_number}"})

@app.route("/instances/all",methods=["GET"])
@cross_origin()
def get_all_instances():
    instancess = []
    count = None
    for k,v in instances.items():
        for k2,v2 in v.items():
           print(v2)
           ben = v2
           count = ben.get_instances_number()
           for i in range (count):
               instance = ben.get_instance(i)
               instancess.append({
                   "jobs" : k,
                   "machines" :k2,
                   "id" : instance.id,
                   "instance" : instance.np_array.tolist()
               }) 
    return jsonify({
            "error" : False, 
            "count" : count,
            "instances" :instancess
        })

if __name__ == '__main__':
    app.debug = True
    app.run()