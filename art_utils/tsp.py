from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from numba import jit, prange
import numpy as np

from art_utils.math import distance

# Traveling salesman ORTools code

@jit(nopython=True, parallel=True)
def distance_matrix(points):
    num_points = len(points)
    out = np.zeros((num_points, num_points))

    for i in prange(num_points):
        for j in prange(num_points):
            out[i, j] = distance(points[i, 0],
                                 points[i, 1],
                                 points[j, 0],
                                 points[j, 1])

    return out


def print_tsp_solution(manager, routing, assignment):
    """Prints assignment on console."""
    print('Objective: {}'.format(assignment.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Objective: {}m\n'.format(route_distance)


def get_tsp_solution(start_node, points, manager, routing, assignment):
    route = []
    index = routing.Start(start_node)

    while not routing.IsEnd(index):
        route.append(points[index])
        previous_index = index
        index = assignment.Value(routing.NextVar(index))

    return route


def solve_tsp_route(points, start_node=0, max_seconds=600):
    dist_matrix = distance_matrix(np.array(points))
    # Convert to integers because ORTools wants that
    dist_matrix = (dist_matrix * 100).astype(int).tolist()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(points), 1, 0)

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = max_seconds  # nice
    # search_parameters.log_search = True

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    return get_tsp_solution(start_node, points, manager, routing, assignment)
