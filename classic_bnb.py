import cplex
import numpy as np
from numba import *
from tqdm import tqdm
from datetime import datetime
import time
import utils_for_cplex as utils

class Node():
    def __init__(self, parent, ub,fix_value, zero_value,candidates_len, graph, obj, colnames, rows, senses, rhs, rownames):
        self.parent = parent
        self.obj = obj # objective, always constant
        self.colnames = colnames # names of variables, always constant
        self.rows = rows # left parts of constraints
        self.senses = senses # types of constraints
        self.rhs = rhs # right parts of constraints
        self.rownames = rownames # names of constraints
        self.children = []
        self.any_solution = None
        self.ub = ub
        self.fix_value = fix_value
        self.zero_value = zero_value
        self.graph = graph
        self.candidates_len = candidates_len

    def solve(self, current_best_integer_solution):
        prob = cplex.Cplex()
        prob.set_results_stream(None)
        prob.objective.set_sense(prob.objective.sense.maximize)
        prob.variables.add(obj=self.obj, names=self.colnames)
        prob.linear_constraints.add(lin_expr=self.rows, senses=self.senses,
                                    rhs=self.rhs, names=self.rownames)
        self.prob = prob
        prob.solve()
        if prob.solution.status[prob.solution.get_status()] != 'optimal':
            print('Not opt')
            raise AssertionError('The solution is not optimal!')

        solution = prob.solution.get_objective_value()
        values = prob.solution.get_values()
        self.values = values
        self.any_solution = solution

        if solution == max_possible_solution: # the hope not dies
            print('Found best possible solution!')
            print("Solution value = ", solution)
            print('Optimal variables: ', dict(zip(self.colnames, values)))
            exit()

        if solution <= current_best_integer_solution:
            return 0, 0 # no children needed

        if np.sum(abs(values - np.array(values).astype(int))) == 0 and solution > current_best_integer_solution:
            print('Found better solution: ', solution)
            current_best_integer_solution = solution
            best_values = values
            return solution, values
        if self.fix_value:
            self.graph = utils.prune_graph_fix_node(self.graph, self.fix_value)
            self.candidates_len += 1
        if self.zero_value:
            self.graph = utils.prune_graph_zero_node(self.graph, self.zero_value)
        color_mask, self.color_num = utils.color_graph(self.graph, n)
        self.ub = self.candidates_len + self.color_num
        self.ub = min(self.ub, self.any_solution)
        if self.ub <= current_best_integer_solution:
            return 0, 0
        for i in range(len(values)):
            if values[i] == int(values[i]):
                continue # everything's ok, no need to branch
            else:
                a = np.zeros(n)
                a[i] = 1
                self.children.append(Node(self,self.ub, i, None,self.candidates_len + 1, self.graph, self.obj, self.colnames, self.rows+[[colnames, a]],
                                        self.senses+'G', self.rhs+[int(values[i])+1], self.rownames+['d'+str(int(self.rownames[-1][1:])+1)]
                ))
                if int(values[i]) != 0:
                    self.children.append(Node(self,self.ub, None, i,self.candidates_len, self.graph, self.obj, self.colnames, self.rows+[[colnames,a]],
                                            self.senses+'L', self.rhs+[int(values[i])], self.rownames+['d'+str(int(self.rownames[-1][1:])+1)]
                    ))
        return solution, values


if __name__ == '__main__':
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/playground.clq')
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/C125.9.clq')
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/hamming6-2.clq')
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/johnson8-2-4.clq')
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/johnson8-4-4.clq')
    graph, m, n = utils.read_graph('./instances/MANN_a9.clq')

    # these graphs require academic edition of cplex
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/johnson16-2-4.clq')
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/hamming8-4.clq')
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/MANN_a27.clq')
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/keller4.clq')
    # graph, m, n = utils.read_graph('./DIMACS_all_ascii/brock200_1.clq')

    prob = cplex.Cplex()
    prob.set_results_stream(None)

    prob.objective.set_sense(prob.objective.sense.maximize)
    obj = np.ones(n)
    colnames = ['x'+str(i) for i in range(n)]
    prob.variables.add(obj=obj, names=colnames)

    rhs1 = np.ones(n).tolist() #right parts of constraints
    senses1 = 'L' * n #all constraints are '<='
    rownames1 = ['b'+str(i+1) for i in range(n)]#names of constraints
    rows1 = []
    for i in range(n):
        a = np.zeros(n)
        a [i] = 1
        rows1.append([colnames, a.tolist()])
    full_dense_graph = np.empty(((n-1)*(n-1) - (n-2)*(n-1)//2, 2))
    for i in range(n):
        s = int((i-1)*(n) - (i-1)*i/2 + n-i) # position counted using arithmetic progression
        full_dense_graph[s:s + n-i-1, 1] = i+1
        full_dense_graph[s:s + n-i-1, 0] = np.linspace(i+2, n, n-i-1)
    full_dense_graph = np.array(full_dense_graph).astype(int)
    for i in range(len(graph)):
        a = graph[i][0]
        b = graph[i][1]
        full_dense_graph[int((b-1)*(n) - (b-2)*(b+1)/2 - 2 + a - b)] = 0
    complemenatary_graph = full_dense_graph[np.sum(full_dense_graph, axis = 1)!=0, :]

    rows2 = []
    for i in range(len(complemenatary_graph)):
        pair = complemenatary_graph[i]
        row = np.zeros(n)
        row[pair[0]-1] = 1
        row[pair[1]-1] = 1
        rows2.append([colnames, row])
    senses2 = 'L'*len(complemenatary_graph)
    rhs2 = np.ones(len(complemenatary_graph)).tolist()
    rownames2 = ['c'+str(i+1) for i in range(len(complemenatary_graph))]#names of constraints

    parent_node = Node(None, 0, None,None,0, graph, obj, colnames, rows1 + rows2, senses1 + senses2, rhs1+rhs2, rownames1 + rownames2)

    current_best_integer_solution = 0
    max_possible_solution = 0
    solution, values = parent_node.solve(current_best_integer_solution)
    max_possible_solution = solution
    if np.sum(abs(values - np.array(values).astype(int))) == 0:
        print('Already integer solution for such data!')
        print("Solution value = ", solution)
        print('Optimal variables: ', dict(zip(parent_node.colnames, values)))
        exit()
    print('Float solution: ', solution)
    node = parent_node
    first_layer_size = len(parent_node.children)
    max_tree_depth = 0
    current_depth_position = 0
    current_progress = 1 - len(parent_node.children)/first_layer_size
    print(datetime.now(), 'Current progress: ', current_progress, ', best solution', current_best_integer_solution, ', max tree depth on last stage: ', max_tree_depth)

    while len(parent_node.children) != 0:
        while(len(node.children) != 0): #go down one branch
            node = node.children[0]
            current_depth_position+=1
            solution, values = node.solve(current_best_integer_solution)
        if current_depth_position > max_tree_depth:
            max_tree_depth = current_depth_position
        if solution > current_best_integer_solution:
            current_best_integer_solution = solution
            current_best_values = values
        while node.any_solution <= current_best_integer_solution or node.ub <= current_best_integer_solution or len(node.children) == 0:
            node.children = []
            node = node.parent
            current_depth_position-=1
        node.children.pop(0)

        while len(node.children) == 0 and node!=parent_node: # return up the branch until a good possibility exists
            node = node.parent
            current_depth_position-=1
            node.children.pop(0) # have just been there

        if 1 - len(parent_node.children)/first_layer_size != current_progress:
            current_progress = 1 - len(parent_node.children)/first_layer_size
            print(datetime.now(), 'Current progress: ', current_progress, ', best solution', current_best_integer_solution, ', max tree depth on last stage: ', max_tree_depth)
            max_tree_depth = 0

        if current_depth_position == 0 and len(node.children)==0:
            print('The search is finished!')
            print('Final solution: ', current_best_integer_solution)
            print('Optimal values: ', dict(zip(colnames, current_best_values)))
            exit()
