import tsplib95
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import timeit
from time import time


def convert(list_to_tuple):
    return tuple(list_to_tuple)


def tabu_search(problem, tabu_size=1000):
    max_time = 60 * 10
    end_time = time() + max_time

    starting_solution = random_solve(problem)
    ending_cost = get_cost(problem, starting_solution)
    ending_solution = starting_solution

    tabu = dict()
    counter = tabu_size
    tabu[tuple(ending_solution)] = (counter, ending_cost)
    current_solution = ending_solution.copy()
    current_solution_cost = ending_cost

    while time() < end_time:
        surrounding = invert(current_solution)  # generujemy otoczenie za pomoca invert
        neighbour_best_solution = None
        neighbour_best_cost = np.inf
        for neighbour in surrounding:
            if tuple(neighbour) not in tabu.keys() or tabu[tuple(neighbour)][0] < 0:
                neighbour_cost = get_cost(problem, neighbour)  # oblicz funkcje celu sasiada
                if neighbour_cost < neighbour_best_cost:  # zamien jesli nowy sasiad ma mniejsza funkcje celu
                    neighbour_best_solution = neighbour
                    neighbour_best_cost = neighbour_cost

        current_solution = neighbour_best_solution
        current_solution_cost = neighbour_best_cost
        tabu[tuple(current_solution)] = (counter, current_solution_cost)

        for key, value in list(tabu.items()):
            temp_counter, temp_cost = value
            tabu[key] = (temp_counter - 1, temp_cost)

        if current_solution_cost < ending_cost:
            ending_cost = current_solution_cost
            ending_solution = current_solution

    for key, value in tabu.items():
        print(key, ':', value, ':', get_cost(problem, key))

    print('Tabu search cost:')
    return get_cost(problem, ending_solution)


"""
def tabu_search(problem, tabu_size=1000):
    max_time = 15
    end_time = time() + max_time
    print('starting time: ')
    starting_solution = calculate_nearest_neighbour(problem, 1)
    tabu = {}
    ending_solution = starting_solution
    counter = 0

    while time() < end_time:
        counter = counter + 1
        surrounding = invert(ending_solution)  # generujemy otoczenie za pomoca invert
        current_solution = None
        current_solution_cost = get_cost(problem, ending_solution)

        converted_surrounding = convert(surrounding)
        for neighbour in (n for n in converted_surrounding if n not in tabu.keys()):
            cost = get_cost(problem, neighbour)
            if cost < current_solution_cost:
                current_solution = neighbour
                current_solution_cost = cost
        if current_solution is None:
            continue
        ending_solution = current_solution
        tabu[current_solution] = (counter, current_solution_cost)
        tabu = {k: (j, val) for k, (j, val) in tabu.items() if counter - tabu_size < j}

    print('ending time: ')

    return get_cost(problem, ending_solution)

"""


def invert(solution: list):
    neighbours = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbours.append(solution[:i] + solution[i:j + 1][::-1] + solution[j + 1:])
    return neighbours


def two_opt(problem):
    solution = random_solve(problem)
    solution_cost = get_cost(problem, solution)
    tabu = {'1': '3000', '3': '3400'}
    can_find_better_solution = True
    while can_find_better_solution:
        surrounding = invert(
            solution)  # surrounding to lista wszystkich rozwiazan do jakich mozna dojsc odwracajac ciag
        new_solution = surrounding[0]  # inicjalizujemy new_solution
        new_solution_cost = get_cost(problem, new_solution)
        for possible_new_solution in surrounding:
            current_cost = get_cost(problem, possible_new_solution)
            if current_cost < new_solution_cost:  # weights decrease
                new_solution_cost = current_cost
                new_solution = possible_new_solution
        if new_solution_cost >= solution_cost:  # nie moze znalezc lepszego rozwiazania
            can_find_better_solution = False
        else:
            solution = new_solution
            solution_cost = new_solution_cost
    return {"solution": solution,
            "cost": solution_cost}


""" an objective function """


def get_cost(problem: tsplib95.models.StandardProblem, tour):
    cost = 0
    for i in range(len(tour) - 1):
        cost += problem.get_weight(*(tour[i], tour[i + 1]))
    cost += problem.get_weight(*(tour[-1], tour[0]))
    return cost


""" a function to generate a random instance of the problem """


def random_solve(problem):
    nodes = list(problem.get_nodes())
    np.random.shuffle(nodes)
    problem.tours.append(nodes)
    return nodes


""" a function to draw a route for a Euclidean instance """


def print_tour(problem, tour_n=0, weights=False):
    g = nx.Graph()
    pos = None

    if problem.edge_weight_type == 'EUC_2D':
        for node in list(problem.get_nodes()):
            g.add_node(node, coord=problem.node_coords[node])
        pos = nx.get_node_attributes(g, 'coord')
    else:
        g.add_nodes_from(list(problem.get_nodes()))
        pos = nx.spring_layout(g, seed=225)

    tour = problem.tours[tour_n]
    for i in range(0, len(tour) - 1):
        g.add_edge(tour[i], tour[i + 1],
                   weight=problem.get_weight(tour[i], tour[i + 1]))
    g.add_edge(tour[-1], tour[0], weight=problem.get_weight(tour[-1], tour[0]))

    nx.draw(g, pos)
    if weights:
        labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    plt.show()


""" algorithm k-random """


def k_random(problem, k):
    if k < 1:
        return
    cost = np.inf  # infinity
    tour1 = []
    current_tour = []
    current_cost = 0
    for _ in range(k):  # _ as a variable; k iterations
        current_tour = np.array(list(problem.get_nodes()))  # node is the index of the city
        np.random.shuffle(current_tour)  # change indexes of the cities in the list
        current_cost = get_cost(problem, current_tour)
        if cost > current_cost:  # weights decrease
            cost = current_cost
            tour1 = current_tour

    problem.tours.append(tour1)

    return cost


""" nearest neighbour algorithm """


def calculate_nearest_neighbour(problem, x):
    not_visited_nodes = list(problem.get_nodes())
    # print(not_visited_nodes)

    if x not in problem.get_nodes():
        return

    not_visited_nodes.remove(x)
    tour = [x]

    while len(not_visited_nodes) != 0:
        cost = np.inf
        next_x = 0

        for i in not_visited_nodes:
            cur_cost = problem.get_weight(*(x, i))

            if cost > cur_cost:
                cost = cur_cost
                next_x = i

        not_visited_nodes.remove(next_x)
        tour.append(next_x)
        x = next_x

    problem.tours.append(tour)  # dostajemy liste z jedna trasa

    # print(problem.get_weight)
    return tour


""" function that prints the weight for the result obtained using the nearest neighbor algorithm """


def nearest_neighbour_get_results(problem, x):
    tour = calculate_nearest_neighbour(problem, x)
    # print(tour)

    return problem.trace_tours([tour])[0]


""" repetitive nearest neighbour algorithm """


def repetitive_nearest_neighbour_get_results(problem):
    tour = []
    weights = []
    for i in problem.get_nodes():
        tour = calculate_nearest_neighbour(problem, i)
        weights.append(problem.trace_tours([tour])[0])

    return min(weights)


def main():
    problem = tsplib95.load(
        '/Users/wiktoriapazdzierniak/Documents/Studia /4_SEM/Algorytmy metaheurystyczne/Zajecia_1/Data/bays29.tsp')

    problem_opt = tsplib95.load(
        '/Users/wiktoriapazdzierniak/Documents/Studia /4_SEM/Algorytmy metaheurystyczne/Zajecia_1/Data/bays29.opt.tour')

    # random_solve(problem)

    # compare_time(problem)
    """
    print('k_random:')
    print(k_random(problem, 25))
    print('NN:')
    print(nearest_neighbour_get_results(problem, 1))

    print('RNN:')
    print(repetitive_nearest_neighbour_get_results(problem))
    """

    print('two_opt:')
    print(two_opt(problem))
    # print_tour(problem)
    print('optimal:')
    print(get_cost(problem, problem_opt.tours[0]))
    print('Tabu search cost:')
    print(tabu_search(problem))


def compare_time(problem2):
    tab = []
    times = []

    difference_1 = 0
    start = timeit.default_timer()
    print('Nearest neighbour cost: ', nearest_neighbour_get_results(problem2, 1))
    stop = timeit.default_timer()
    difference = stop - start
    print('Time of the nearest neighbour: ', str(difference))
    k = 100

    for i in range(1, k + 1):
        start = timeit.default_timer()
        k_random(problem2, i)
        stop = timeit.default_timer()
        difference_1 = stop - start
        times.append(difference_1)
    for i in range(0, len(times) - 1):
        tab.append(abs(difference - times[i]))

    print('Time difference: ', min(tab), ' k: ', tab.index(min(tab)))
    print('k-random cost: ', k_random(problem2, tab.index(min(tab))))


if __name__ == '__main__':
    main()
