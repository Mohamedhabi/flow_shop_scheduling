from utils import Instance

def get_results(instance):
    """Get the results of the algorithm on the specified instance

    Args:
        instance (class:instance): an FPS instance

    Returns:
        list: jobs order
    """

    #exemples
    instance.get_cost(0,0) # The cost of Job 0, on machine 0
    instance.get_cost_on_machines(0) # the costs of job 0 on all machines
    instance.get_jobs(0) # Get the costs of all jobs on machine 0
    #...
    return {
        "C_max": 13,
        "order":list(range(instance.get_jobs_number()))
        }