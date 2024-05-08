import tkinter as tk
from tkinter import ttk, colorchooser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import pandas as pd
from collections import Counter
import community as community_louvain
from networkx.algorithms.community import girvan_newman
import community.community_louvain
import tkinter.messagebox as messagebox
from networkx.algorithms.community.quality import modularity
import pydot
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.metrics import normalized_mutual_info_score
from cdlib import evaluation
import igraph as ig
import numpy as np
from cdlib import algorithms, evaluation



def apply_louvain(G):
    partition = community.best_partition(G)

    return partition


# Function to apply Girvan Newman algorithm
def girvanstandard(G):
    newman = nx.community.girvan_newman(G)
    for x in range(3):
        level3 = next(newman)
    return level3



def apply_girvan_newman(G):
    level3=girvanstandard(G)

    sorted_communities_tuple = tuple()

    # sort each community add them in tuple of communities
    for community in level3:
        sorted_community = sorted(community)
        sorted_communities_tuple += (sorted_community,)
    return sorted_communities_tuple

# Functions to calculate modularity score
def calculatemodtonewman(G):
    level3 = girvanstandard(G)
    communities_newman = []
    #list of lists
    for community_set in level3:
        community_list = list(community_set)
        # Append the community list to the main list of communities
        communities_newman.append(community_list)

    # Calculate modularity using the Newman-Girvan communities
    modularity_score_newman = nx.community.modularity(G, communities_newman)

    return modularity_score_newman

def calculatemodtolouvain(G):
    partition = apply_louvain(G)
    communities = []
    unique = set(partition.values())
    # Iterate over each unique community ID
    # list of lists
    for community_id in unique:
        members = []
        # Iterate over all nodes in the partition
        for node in partition.keys():
            # If this node belongs to the current community add it -_-
            if partition[node] == community_id:
                members.append(node)
        # After collecting all nodes of this community collect them here aaaaaaaaaaa
        communities.append(members)
    modularity_score = nx.community.modularity(G, communities)
    return modularity_score


def Normalizemutualnformation_louvain(G):
    partition = apply_louvain(G)
    ground_truth_labels = []
    for node in G.nodes():
        # Append the 'classS' attribute of each node to the list
        ground_truth_label = G.nodes[node]['classS']
        ground_truth_labels.append(ground_truth_label)

    predicted_labels = []
    for node in G.nodes():
        predicted_label = partition[node]
        predicted_labels.append(predicted_label)


    nmi_score = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
    return nmi_score

def Normalizemutualnformation_newman(G):
    level3 = girvanstandard(G)
    community_labels = {}
    #LIKE PARTITIONS ("1534":0 )and so on
    for i, community in enumerate(level3):
        for node in community:
            community_labels[node] = i

    ground_truth_labels = []
    for node in G.nodes():
        # Append the 'classS' attribute of each node to the list
        ground_truth_label = G.nodes[node]['classS']
        ground_truth_labels.append(ground_truth_label)

    predicted_labels = []
    for node in G.nodes():
        predicted_label = community_labels[node]
        predicted_labels.append(predicted_label)


    nmi_score = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
    return  nmi_score


def calculate_conductance(G):
    partition = apply_louvain(G)
    # Convert partition format to list of node sets
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = set()
        communities[comm_id].add(node)

    # Calculate conductance for each community
    conductances = {}
    for comm_id, nodes in communities.items():
        # Conductance calculation
        conductance = nx.conductance(G, nodes)
        conductances[comm_id] = conductance

    return conductances


def calculate_conductance_for_newman(G):
    communities =algorithms.girvan_newman(G,level=3)   # This function must return the community sets
    conductances = {}
    # Enumerate through each community and calculate conductance
    conductance_value = evaluation.conductance(G, communities, summary=False)
    conductance_dict = {}
    for i, score in enumerate(conductance_value):
        conductance_dict[i] = score

    return conductance_dict



def calculate_pagerank(G):
    return nx.pagerank(G)

def calculate_betweenness_centrality(G):
    return nx.betweenness_centrality(G)

def filtergraphcentrality(G,v):
    filtered_nodes = []
    degree_centrality_dict = nx.degree_centrality(G)

    for node, centrality in degree_centrality_dict.items():
        if centrality >= v:
            filtered_nodes.append(node)
    return G.subgraph(filtered_nodes)

def filtergraphclosness(G,v):
    filterd_graph=[]
    closness_results=nx.closeness_centrality(G)

    for node,center in closness_results.items():
        if center >= v:
            filterd_graph.append(node)

    return G.subgraph(filterd_graph)

def filtergraphbetweenness(G,v):
    filtered_graph=[]
    betweenness_results=nx.betweenness_centrality(G)

    for node,center in  betweenness_results.items():
        if center >= v:
            filtered_graph.append(node)

    return G.subgraph(filtered_graph)

def filtergraphcentralityRange(G,range1,range2):
    filtered_nodes = []
    degree_centrality_dict = nx.degree_centrality(G)

    for node, centrality  in degree_centrality_dict.items():
        if range1 <= centrality <= range2:
            filtered_nodes.append(node)
    return G.subgraph(filtered_nodes)

def filtergraphclosnessrange(G,range1,range2):
    filterd_graph=[]
    closness_results=nx.closeness_centrality(G)

    for node,center in closness_results.items():
        if range1 <= center <= range2:
            filterd_graph.append(node)

    return G.subgraph(filterd_graph)

def filtergraphbetweennessrange(G,range1,range2):
    filtered_graph=[]
    betweenness_results=nx.betweenness_centrality(G)

    for node,center in  betweenness_results.items():
        if range1 <= center <= range2:
            filtered_graph.append(node)

    return G.subgraph(filtered_graph)

def RB_HERE_for_directed_graph(G):
    coms = algorithms.rb_pots(G)
    return coms



#handle diricted graphs
def RB_Modularity(G,communities):

    m = G.number_of_edges()  # Total number of edges in the network
    modularity = 0
    #l_c total number of edges within the community.
    #k_in_c and k_out_c, the sum of in-degrees and out-degrees for the nodes

    for community in communities:
        L_c = sum(1 for i in community for j in community if G.has_edge(i, j))
        k_in_c = sum(G.in_degree(i) for i in community)
        k_out_c = sum(G.out_degree(j) for j in community)
        modularity += (L_c / m) - (k_in_c * k_out_c / (m * m))


    return modularity

def Normalizemutualnformation_RB(G):
    coms=RB_HERE_for_directed_graph(G)
    ground_truth_labels = []
    for node in G.nodes():
        # Append the 'classS' attribute of each node to the list
        ground_truth_label = G.nodes[node]['Class']
        ground_truth_labels.append(ground_truth_label)

    predicted_labels = []
    for node in G.nodes():
        for idx, community in enumerate(coms.communities):
            if node in community:
                predicted_labels.append(idx)
                break


    nmi_score = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
    return nmi_score

def calculate_conductance_for_RB(G):
    coms = algorithms.rb_pots(G)  # This function must return the community sets
    conductance_value = evaluation.conductance(G,coms,summary=False)
    conductance_dict = {}
    for i, score in enumerate(conductance_value):
        conductance_dict[i] = score

    return conductance_dict



def part_class(G):
    communities = {}
    for node, data in G.nodes(data=True):
        node_class = data.get('classS')  # Default to 'Unknown' if class is not specified
        if node_class not in communities:
            communities[node_class] = set()
        communities[node_class].add(node)

    return communities

def part_gender(G):
    communities = {}
    for node, data in G.nodes(data=True):
        node_class = data.get('gender')  # Default to 'Unknown' if class is not specified
        if node_class not in communities:
            communities[node_class] = set()
        communities[node_class].add(node)

    return communities
