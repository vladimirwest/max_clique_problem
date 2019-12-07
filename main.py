import cplex
import numpy as np
from termcolor import colored
from numba import *
from tqdm import tqdm
from datetime import datetime
import time
import utils as utils
import argparse
import numpy as np
import numba
import networkx as nx

def read_networkx_graph(file_path):
    '''
        Parse .col file and return graph object
    '''
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            # if line.startswith('c'):  # graph description
            #     print(*line.split()[1:])
            # first line: p name num_of_vertices num_of_edges
            if line.startswith('p'):
                p, name, vertices_num, edges_num = line.split()
                # print('{0} {1} {2}'.format(name, vertices_num, edges_num))
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                edges.append((v1, v2))
            else:
                continue
        return nx.Graph(edges), vertices_num

def greedy_coloring_heuristic(graph):
    '''
    Greedy graph coloring heuristic with degree order rule
    '''
    color_num = iter(range(0, len(graph)))
    color_map = {}
    used_colors = set()
    nodes = [node[0] for node in sorted(nx.degree(graph),
                                        key=lambda x: x[1], reverse=True)]
    color_map[nodes.pop(0)] = next(color_num)  # color node with color code
    used_colors = {i for i in color_map.values()}
    while len(nodes) != 0:
        node = nodes.pop(0)
        neighbors_colors = {color_map[neighbor] for neighbor in
                            list(filter(lambda x: x in color_map, graph.neighbors(node)))}
        if len(neighbors_colors) == len(used_colors):
            color = next(color_num)
            used_colors.add(color)
            color_map[node] = color
        else:
            color_map[node] = next(iter(used_colors - neighbors_colors))
    return len(used_colors), color_map

def get_neighbours_graph(graph, vertex):
    neighbours = []
    for neighbour in graph.neighbors(str(vertex)):
        neighbours.append(neighbour)
    return graph.subgraph(neighbours)

def get_sorted_colors(color_map):
    colors_dict = {}
    for item in color_map.items():
        if item[1] in colors_dict:
            colors_dict[item[1]] += 1
        else:
            colors_dict[item[1]] = 1
    sorted_colors = sorted(colors_dict.items(), key=lambda x:x[1])
    return sorted_colors


class Node():
    def __init__(self, parent,fix_vertex, candidate_graph, ub, clique):
        self.parent = parent
        self.children = []
        self.fix_vertex = fix_vertex
        self.candidate_graph = candidate_graph
        self.ub = ub
        self.clique = clique

    def solve(self, best_known_solution):
        if self.ub <= best_known_solution:
            return best_known_solution
        if self.fix_vertex:
            self.candidate_graph = get_neighbours_graph(self.candidate_graph, self.fix_vertex)
        if self.parent:
            for bro in self.parent.children:
                gr = bro.candidate_graph.copy()
                if self.fix_vertex in gr.nodes:
                    gr.remove_node(self.fix_vertex)
                    bro.candidate_graph = gr
        if len(self.candidate_graph.nodes) == 0:
            if len(self.clique) > best_known_solution:
                print('Found better solution:', len(self.clique))
                best_known_solution = len(self.clique)
                return best_known_solution
            else:
                return best_known_solution
        self.color_num, color_map = greedy_coloring_heuristic(self.candidate_graph)

        self.ub = (len(self.clique) + self.color_num)

        if self.ub <= best_known_solution:
            return best_known_solution
        sorted_colors = get_sorted_colors(color_map)
        children = []
        for i in sorted_colors:
            color = i[0]
            children.append([])
            for node in color_map:
                if color_map[node] == color:
                    children[-1].append(node)
        for i in range(len(children)):
            for j in range(len(children[i])):
                node = children[i][j]
                if len(children[i]) == 1:
                    self.children.append(Node(parent = self, fix_vertex = node, candidate_graph = self.candidate_graph, ub = self.ub - 1, clique = self.clique + [node]))
                else:
                    self.children.append(Node(parent = self, fix_vertex = node, candidate_graph = self.candidate_graph, ub = self.ub, clique = self.clique + [node]))
        return best_known_solution


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph',  type=str, default = 'p')
    args = parser.parse_args()
    if args.graph == 'play':
        graph, n = read_networkx_graph('./instances/playground.clq')
    elif args.graph == 'MANN_a9':
        graph, n = read_networkx_graph('./instances/MANN_a9.clq')
    elif args.graph == 'keller4':
        graph, n = read_networkx_graph('./instances/keller4.clq')
    elif args.graph == 'hamming6':
        graph, n = read_networkx_graph('./instances/hamming6-2.clq')
    elif args.graph == 'brock200_1':
        graph, n = read_networkx_graph('./instances/brock200_1.clq')

    n = int(n)
    best_known_solution = 0
    parent_node = Node(parent = None, fix_vertex = None, candidate_graph = graph, ub = n, clique = [])
    parent_node.solve(best_known_solution)

    first_layer_size = len(parent_node.children)
    current_progress = 1 - len(parent_node.children)/first_layer_size
    print(datetime.now(), 'Current progress: ', current_progress, ', best solution', best_known_solution)
    node = parent_node
    depth = 0
    max_depth = 0
    stamp_count = 0

    t = time.time()
    while len(parent_node.children) != 0:
        while(len(node.children) != 0): #go down one branch
            node = node.children[0]
            best_known_solution = node.solve(best_known_solution)
            depth += 1
        if best_known_solution < len(node.clique):
            best_known_solution = len(node.clique)
            print(colored((node.children, node.clique) ,'yellow'))
            print('Found better', best_known_solution)
        if depth > max_depth:
            max_depth = depth


        while (node.ub <= best_known_solution and node != parent_node) or (node.children == [] and node != parent_node):
            node.children = []
            node = node.parent
            depth -= 1

        for ch in node.children[1:]:
            if ch.children != []:
                gr = ch.candidate_graph.copy()
                gr.remove_node(node.children[0].fix_vertex)
                ch.candidate_graph = gr
        if node == parent_node:
            current_progress = 1 - len(parent_node.children)/first_layer_size
            print(datetime.now(), 'Current progress: ', current_progress, ', best solution', best_known_solution)
        node.children.pop(0)
        if node == parent_node and len(node.children)==0:
            print('The search is finished!')
            print('Final solution: ', best_known_solution)
            exit()
