"""
cluster.py
"""
import sys
import networkx as nx
from collections import Counter
import pickle
import matplotlib.pyplot as plt # for debugging
import math
from collections import Counter, defaultdict, deque
import copy

def create_graph(users, friends_dict):
    """ 
    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    list_friend = []
    edges = []
    #list_friend = [i for i in friend_counts if friend_counts[i]>1]
    graph = nx.Graph()

    for u in users:
        screen_name_id = u['id']
        screen_name = u['screen_name']
        graph.add_node(screen_name_id)
        #f = set(list_friend) & set(friends_dict[screen_name])
        f = set(friends_dict[screen_name]['ids'])
        for i in f:
            tup = (screen_name_id, i)
            edges.append(tup)

    graph.add_edges_from(edges)
    return graph

def draw_network(graph, users, filename):
    """
        for debugging the graph
    """
    label = {n:n for n in graph.nodes()}
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(graph, node_color='r', labels=label, width=.1, node_size=100)
    plt.axis("off")
    #plt.savefig(filename)
    plt.show()

def partition_girvan_newman(graph, max_depth):
    graph_c = graph.copy()
    ret_list = []
    #ibet_dict = approximate_betweenness(graph_c, max_depth)
    ibet_dict = nx.betweenness_centrality(graph)
    ib = sorted(ibet_dict.items(), key=lambda i: i[1], reverse=True)
    components = [c for c in nx.connected_component_subgraphs(graph_c)]
    while len(components) == 1:
        graph_c.remove_edge(*ib[0][0])
        del ib[0]
        components = [c for c in nx.connected_component_subgraphs(graph_c)]
    for c in components:
        ret_list.append(c)

    return ret_list
        
def main():
    f = open('./data/users.txt','rb')
    f2 = open('./data/friends.txt', 'rb')
    users = pickle.load(f)
    #user_list = [u['screen_name'] for u in users]
    # Creating the graph
    friends_dict = pickle.load(f2)
    graph = create_graph(users, friends_dict)
    #print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    #draw_network(graph, user_list, 'network.png')
    # begin clustering
    clusters = partition_girvan_newman(graph, math.inf)
    #print(clusters)
    total_nodes = 0
    for c in clusters:
        total_nodes += c.number_of_nodes()

    list_of_summarize = []
    list_of_summarize.append(len(clusters))
    list_of_summarize.append(total_nodes / len(clusters))
    
    f4 = open('./data/sum.txt','ab')
    pickle.dump(list_of_summarize, f4)
if __name__ == '__main__':
    main()
