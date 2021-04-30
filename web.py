import flask
from flask import json,request

app = flask.Flask(__name__)

app.config["DEBUG"] = True

algorithms = [
    {
        "name"  : "Branch and Bound",
        "id" : 0
    }
]
instances = [
    {
        'jobs'  : 3,
        'machines' : 4,
        'costs' : [
            [1,3,4,6], 
            [2,3,1,5], 
            [3,4,5,4]
        ]
    }, 
    {
        'jobs'  : 2,
        'machines' : 3,
        'costs' : [
            [1,3,4], 
            [2,3,1], 
        ]
    }, 
    ]

@app.route('/',methods=["GET"])
def home():
    return 'Hello world'

@app.route('/api/algorithms/all',methods=["GET"])
def get_all_algorithms():
    return json.jsonify(algorithms)

@app.route('/api/algorithms',methods=["GET"])
def get_algorithm_by_id():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "No instance ID field provided"
    result = algorithms[id]
    return json.jsonify(result)

@app.route('/api/instances/all',methods=['GET'])
def get_all_instances():
    return json.jsonify(instances)

@app.route('/api/instances',methods=['GET'])
def get_instance_by_id():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "No instance ID field provided"
    result = instances[id]
    return json.jsonify(result)    

# launching the backend api
app.run()