from utils import Instance
from core.methods import *
def execute(method_id: int,instance: Instance,params:dict):
    if(method_id == BRANCH_AND_BOUND["id"]):
        print(f"executing {BRANCH_AND_BOUND['method']}")
    if(method_id == NEH["id"]):
        print(f"executing {NEH['method']}")
    if(method_id == CDS["id"]):
      print(f"executing {CDS['method']}")
    if(method_id == ACO["id"]):
        print(f"executing {ACO['method']}")
    if(method_id == SA["id"]):
       print(f"executing {SA['method']}")
    if(method_id == GA["id"]):
        print(f"executing {GA['method']}")
    rn = list(range(0,instance.get_jobs_number()))
    print(rn)
    res =  {
        "instance_id" : instance.id,
        "method" : "method",
        "makespan" : 314,
        "execution_time" : 31.13,
        "sequence" : rn, 
        "additional_info" :{
            "leafs" : 1,
            "explored" : 100,
            "pruned" : 300
        }
     }
    return res,None
