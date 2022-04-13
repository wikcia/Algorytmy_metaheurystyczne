import tsplib95
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import timeit


def tabu_search(problem):
    tabu_list = []
    solution = random_solve(problem)  # rozwiazanie poczatkowe pi
    time = 0.015

    x = 1
    while x == 1:
        start = timeit.default_timer()
        solution_cost = get_cost(problem, solution)
        can_find_better_solution = True
        while can_find_better_solution:
            surrounding = invert(          # generowanie otoczenia
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
                if new_solution not in tabu_list:
                    solution = new_solution
                    tabu_list.append(solution)
                    solution_cost = new_solution_cost

        stop = timeit.default_timer()
        difference = stop - start
        print(difference)
        if difference > time:
            x = 0

    print(tabu_list)
    print('Solution cost:')
    print(solution_cost)


def invert(solution: list):
    neighbours = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbours.append(solution[:i] + solution[i:j + 1][::-1] + solution[j + 1:])
    return neighbours


def two_opt(problem):
    solution = random_solve(problem)
    solution_cost = get_cost(problem, solution)
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

    #random_solve(problem)

    #compare_time(problem)
    """
    print('k_random:')
    print(k_random(problem, 25))
    print('NN:')
    print(nearest_neighbour_get_results(problem, 1))

    print('RNN:')
    print(repetitive_nearest_neighbour_get_results(problem))
    """

    #print('two_opt:')
    #print(two_opt(problem))
    #print_tour(problem)
    print('optimal:')
    print(get_cost(problem, problem_opt.tours[0]))

    tabu_search(problem)


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
