import numpy as np
import numba

def read_graph(path):
    file = open(path).readlines()
    edges = []
    data_mode = False
    for i in file:
        if data_mode:
            edges.append(np.fromstring(i[1:], dtype = int, sep = ' '))
        # if i[:17] == 'c Largest Clique:':
        #     largest_clique = int(i[17:])
        if i[:6] == 'p edge':
            n = np.fromstring(i[7:], dtype = int, sep = ' ')[0] # vertex num
            m = np.fromstring(i[7:], dtype = int, sep = ' ')[1] # edges num
            data_mode = True
        if i[:5] == 'p col':
            n = np.fromstring(i[6:], dtype = int, sep = ' ')[0] # vertex num
            m = np.fromstring(i[6:], dtype = int, sep = ' ')[1] # edges num
            data_mode = True
    return np.array(edges), m, n

@numba.njit
def prune_graph_fix_node(graph, value):
    graph0 = graph[graph[:, 0]==value, :]
    if len(graph0) > 0:
        a = graph0[:, 1]
        cum = np.cumsum(np.diff(np.concatenate((np.array([0]), a))) - 1) # recount numbers
        graph0[:, 1] -= cum
        graph0[:, 0] -= cum[-1]
        graph0[:, 0] -= graph0[0,0] - graph0[-1, 1] - 1
    graph1 = graph[graph[:, 1]==value, :]
    if len(graph1) > 0:
        if len(graph0) > 0:
            graph1[:, :] -= (graph1[0, 1] - graph0[-1,0])
        a = graph1[:, 0]
        if len(graph0) > 0:
            cum = np.cumsum(np.diff(np.concatenate((np.array([graph0[-1,0]]), a))) - 1) # recount numbers
        else:
            cum = np.cumsum(np.diff(np.concatenate((np.array([0]), a))) - 1) - 1 # recount numbers
        graph1[:, 0] -= cum
    graph = np.concatenate((graph0, graph1), axis = 0)
    return graph

@numba.njit
def get_full_flaged_graph(graph, n):
    full_dense_graph = np.empty(((n-1)*(n-1) - (n-2)*(n-1)//2, 3), dtype = np.uint8)#0, 1 - edges, 3 - flag
    full_dense_graph[:, 2] = 0
    for i in numba.prange(n):
        s = int((i-1)*(n) - (i-1)*i/2 + n-i) # position counted using arithmetic progression
        full_dense_graph[s:s + n-i-1, 1] = i+1
        full_dense_graph[s:s + n-i-1, 0] = np.linspace(i+2, n, n-i-1)
    for i in numba.prange(len(graph)):
        a = graph[i][0]
        b = graph[i][1]
        full_dense_graph[int((b-1)*(n) - (b-2)*(b+1)/2 - 2 + a - b), 2] = 1
    return full_dense_graph

@numba.njit
def color_graph(graph, n):
    full_dense_graph = get_full_flaged_graph(graph, n)
    color_mask = np.zeros(n, dtype = np.uint8)
    color_num = 1
    for i in numba.prange(n):
        full_neighbours_left = np.empty((2*i//2, 3), dtype= np.uint8)
        for j in range(i):
            full_neighbours_left[j] = full_dense_graph[int(j*n - (j-1)*(j+2)/2 - 2 + i - j), :]
        neighbours_left = full_neighbours_left[full_neighbours_left[:, 2] == 1, 1]
        colors = color_mask[neighbours_left - 1]
        if len(colors) == 0:
            color_mask[i] = 1
        else:
            colors = np.sort(colors)
            colors2 = np.concatenate((np.array([0]), colors), axis = 0)
            for j in range(len(colors2) - 1):
                if colors2[j+1] - colors2[j] > 1:
                    color_mask[i] = colors2[j] +1
                    break
            else:
                color_mask[i] = np.max(colors) + 1
    return color_mask, np.max(color_mask)

@numba.njit
def prune_graph_zero_node(graph, value):
    mask0 = graph[:, 0] > value
    mask1 = graph[:, 1] > value
    graph[mask0, 0] -= 1
    graph[mask1, 1] -= 1
    mask0 = graph[:, 0] != value
    mask1 = graph[:, 1] != value
    graph = graph[mask0&mask1, :]
    return graph


@numba.njit
def get_complementary_graph(graph, n):
    full_dense_graph = np.empty(((n-1)*(n-1) - (n-2)*(n-1)//2, 2), dtype = np.uint8)
    for i in numba.prange(n):
        s = int((i-1)*(n) - (i-1)*i/2 + n-i) # position counted using arithmetic progression
        full_dense_graph[s:s + n-i-1, 1] = i+1
        full_dense_graph[s:s + n-i-1, 0] = np.linspace(i+2, n, n-i-1)
    for i in numba.prange(len(graph)):
        a = graph[i][0]
        b = graph[i][1]
        full_dense_graph[int((b-1)*(n) - (b-2)*(b+1)/2 - 2 + a - b)] = 0
    complementary_graph = full_dense_graph[np.sum(full_dense_graph, axis = 1)!=0, :]
    return complementary_graph
