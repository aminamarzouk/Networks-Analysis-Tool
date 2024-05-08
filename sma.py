import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
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
from graph_functions import *

# Initialize the main window
root = tk.Tk()
root.title('Graph Visualizer')

# Initialize dictionaries
global G
node_ID = {}
node_colors = {}
node_sizes = {}
node_labels = {}
node_class = {}
node_gender = {}
edges = {}

Graph_Input = ttk.LabelFrame(root, text='Graph Input')
Graph_Input.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
Graph_Input.columnconfigure(0, weight=1)
Graph_Input.columnconfigure(1, weight=1)

# Adding graph type options inside the Graph_Input frame
graph_type_var = tk.StringVar(value='u')  # Default to 'undirected'
ttk.Radiobutton(Graph_Input, text="Directed", variable=graph_type_var, value='d').grid(row=0, column=0, padx=10, pady=10, sticky='w')
ttk.Radiobutton(Graph_Input, text="Undirected", variable=graph_type_var, value='u').grid(row=0, column=1, padx=10, pady=10, sticky='w')

# Confirm button to apply graph type selection
confirm_button = ttk.Button(Graph_Input, text='Create Graph', command=lambda: graph_type(graph_type_var.get()))
confirm_button.grid(row=1, column=0, columnspan=2, pady=10)

# Node file selection
node_file_label_var = tk.StringVar()
ttk.Button(Graph_Input, text='Select Nodes CSV', command=lambda: open_file_dialog(node_file_label_var)).grid(row=2, column=1)
ttk.Label(Graph_Input, text='Nodes file').grid(row=2, column=0)
ttk.Label(Graph_Input, textvariable=node_file_label_var).grid(row=3, column=1)

# Edge file selection
edge_file_label_var = tk.StringVar()
ttk.Button(Graph_Input, text='Select Edges CSV', command=lambda: open_file_dialog(edge_file_label_var)).grid(row=4, column=1)
ttk.Label(Graph_Input, text='Edges file').grid(row=4, column=0)
ttk.Label(Graph_Input, textvariable=edge_file_label_var).grid(row=5, column=1)

# Button to draw the graph with selected files
ttk.Button(Graph_Input, text='Draw Graph', command=lambda: draw_graph_with_files(node_file_label_var.get(), edge_file_label_var.get())).grid(row=6, column=1, pady=10)





def graph_type(type_graph):
    global G
    # Check if G is not defined in the global scope
    if 'G' not in globals():
        if type_graph == 'd':
            G = nx.DiGraph()
        else:
            G = nx.Graph()
    else:
        if type_graph == 'd':
            if not G.is_directed():
                G = G.to_directed()
        else:
            if G.is_directed():
                G = G.to_undirected()

    draw_graph(G, layout_entry.get())

def open_file_dialog(label_var):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    label_var.set(file_path)


def draw_graph_with_files(node_file_path, edge_file_path):
    # Load CSV file for nodes
    data_nodes = pd.read_csv(node_file_path)
    data_nodes['ID'] = data_nodes['ID'].astype(str)
    for index, row in data_nodes.iterrows():
        G.add_node(row['ID'],classS=row['Class'],gender=row['Gender'])
        node_class[row['ID']] = row['Class']
        node_gender[row['ID']] = row['Gender']
        node_colors[row['ID']] = '#3B7EC1'  # Default color
        node_sizes[row['ID']] = 200  # Default size
        node_labels[row['ID']] = row['ID']  # Default label

    # Load CSV file for edges
    data_edges = pd.read_csv(edge_file_path)
    data_edges['Source'] = data_edges['Source'].astype(str)
    data_edges['Target'] = data_edges['Target'].astype(str)
    for index, row in data_edges.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row.get('weight', 1))
        edges[row['Source']] = row['Target']
    draw_graph(G, layout_entry.get())


x = 'o'
y = 'black'


# Function to draw the graph
def draw_graph(G, layout):
    plt.clf()
    ax = plt.gca()
    ax.clear()  # Clear current axes

    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G,scale=2)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'tree':
        try:
            pos = nx.nx_agraph.pygraphviz_layout(G, prog="dot")
        except ImportError:
            print("PyGraphviz is required for tree layout")
            return
    elif layout == 'radial':
        pos = nx.nx_agraph.pygraphviz_layout(G, prog='twopi')
    else:
        pos = nx.spring_layout(G,scale=2)

    # Draw the graph using the chosen layout
    nx.draw_networkx(G, pos, with_labels=True,
                     node_color=list(node_colors.values()),
                     node_size=list(node_sizes.values()),
                     edge_color=y,
                     node_shape=x,
                     labels=node_labels)

    canvas.draw()


# Embedding matplotlib in Tkinter
fig = plt.figure(figsize=(12, 9))
canvas = FigureCanvasTkAgg(fig, master=root)
plot_widget = canvas.get_tk_widget()
plot_widget.grid(row=0, column=2, rowspan=5)

# Node attribute controls
node_label_frame = ttk.LabelFrame(root, text='Node Attributes')
node_label_frame.grid(row=2, column=0, sticky='ew')

tk.Label(node_label_frame, text='Node').grid(row=0, column=0)
node_entry = ttk.Entry(node_label_frame)
node_entry.grid(row=0, column=1)

tk.Label(node_label_frame, text='Name').grid(row=1, column=0)
tk.Label(node_label_frame, text='Size').grid(row=2, column=0)
tk.Label(node_label_frame, text='Color').grid(row=3, column=0)

size_button = ttk.Button(node_label_frame, text='Change Name', command=lambda: change_name(node_entry.get()))
size_button.grid(row=1, column=1)
size_button = ttk.Button(node_label_frame, text='Change Size', command=lambda: change_size(node_entry.get()))
size_button.grid(row=2, column=1)
color_button = ttk.Button(node_label_frame, text='Choose Color', command=lambda: choose_color(node_entry.get()))
color_button.grid(row=3, column=1)

tk.Label(node_label_frame, text='Shape').grid(row=4, column=0)
Shape_entry = ttk.Combobox(node_label_frame)

# Set the options for the combo box
Shape_entry['values'] = ('circle', 'square', 'diamond', 'triangle', 'pentagon')
Shape_entry.state(['readonly'])
Shape_entry.grid(row=4, column=1)
Shape_entry.set('circle')

Shape_button = ttk.Button(node_label_frame, text='Choose Shape', command=lambda: choose_Shape(Shape_entry.get()))
Shape_button.grid(row=5, column=1)


def choose_Shape(shape):
    plt.clf()
    ax = plt.gca()
    ax.clear()  # Clear current axes
    global x
    # Choose layout
    if shape == 'circle':
        x = 'o'
        draw_graph(G, layout_entry.get())
    elif shape == 'square':
        x = 's'
        draw_graph(G, layout_entry.get())
    elif shape == 'diamond':
        x = 'D'
        draw_graph(G, layout_entry.get())
    elif shape == 'triangle':
        x = '^'
        draw_graph(G, layout_entry.get())
    elif shape == 'pentagon':
        x = 'p'
        draw_graph(G, layout_entry.get())


def change_name(node):
    # Function to handle the confirmation of the entered size
    def confirm_name(new_name_entry):
        if node in node_labels:
            node_labels[node] = new_name_entry.get()

        draw_graph(G, layout_entry.get())

        name_popup.destroy()  # Close the popup window after confirmation

        # Create a popup window for entering the new size

    name_popup = tk.Toplevel(root)
    name_popup.title("Enter New Name")

    tk.Label(name_popup, text="Enter new label for the node:").grid(row=0, column=0)
    new_name_entry = ttk.Entry(name_popup)
    new_name_entry.grid(row=0, column=1)

    confirm_button = ttk.Button(name_popup, text="Confirm", command=lambda: confirm_name(new_name_entry))
    confirm_button.grid(row=1, columnspan=2)


def change_size(node):
    # Function to handle the confirmation of the entered size
    def confirm_size(new_size_entry):
        new_size = int(new_size_entry.get())

        if node in node_sizes:
            node_sizes[node] = new_size
        draw_graph(G, layout_entry.get())
        size_popup.destroy()  # Close the popup window after confirmation

        # Create a popup window for entering the new size

    size_popup = tk.Toplevel(root)
    size_popup.title("Enter New Size")

    tk.Label(size_popup, text="Enter new size for the node:").grid(row=0, column=0)
    new_size_entry = ttk.Entry(size_popup)
    new_size_entry.grid(row=0, column=1)

    confirm_button = ttk.Button(size_popup, text="Confirm", command=lambda: confirm_size(new_size_entry))
    confirm_button.grid(row=1, columnspan=2)


def choose_color(node):
    global z
    color_code = colorchooser.askcolor(title="Choose color")[1]
    z = color_code
    if node in node_colors:
        node_colors[node] = z

    draw_graph(G, layout_entry.get())


layout_graph = ttk.LabelFrame(root, text='LAYOUT Algo.')
layout_graph.grid(row=4, column=0, sticky='ew')

tk.Label(layout_graph, text='algo').grid(row=0, column=0)
layout_entry = ttk.Combobox(layout_graph)

# Set the options for the combo box
layout_entry['values'] = ('spring', 'circular', 'tree', 'radial')
layout_entry.state(['readonly'])
layout_entry.grid(row=0, column=1)
layout_entry.set('Spring Layout')

tk.Label(layout_graph, text='algo layout').grid(row=1, column=0)
layout_button = ttk.Button(layout_graph, text='Choose layout', command=lambda: draw_graph(G, layout_entry.get()))
layout_button.grid(row=1, column=1)




Edge_label_frame = ttk.LabelFrame(root, text='Edge Attributes')
Edge_label_frame.grid(row=3, column=0, sticky='ew')
color_button2 = ttk.Button(Edge_label_frame, text='Choose color', command=lambda: choose_color2())
color_button2.grid(row=2, column=1)


def choose_color2():
    color_code = colorchooser.askcolor(title="Choose color")[1]
    global y
    y = color_code
    draw_graph(G, layout_entry.get())








# Function to display community detection results
def display_community_results(G, algorithm_name, communities):
    if algorithm_name == "Louvain algorithm":
        result_text = f"Community detection using {algorithm_name}:\n"
        m = 0
        n = 0

        max_community_id=max(communities.values())
        result_text += f"Number of communities:{max_community_id + 1} \n"
        for i in range(max_community_id + 1):
            result_text += f"Community:{i+1}"
            for key in communities:

                if (communities[key] == m):
                    result_text += f" {key}"
            result_text += "\n"
            m += 1
    elif algorithm_name == 'RB algorithm':
        communities1=communities.communities
        num_communities = len(communities1)
        result_text = f"Community detection using {algorithm_name}:\n"
        result_text += f"Number of communities: {num_communities}\n"
        for i, com in enumerate(communities1, 1):
           result_text+= f"Community {i}: {com}\n"

    else:
        num_communities = len(communities)

        result_text = f"Community detection using {algorithm_name}:\n"
        result_text += f"Number of communities: {num_communities}\n"
        for i, community in enumerate(communities, 1):
            result_text += f"Community {i}: {community}\n"
    return result_text


def show_results():
    if(G.is_directed()==False):
        louvain_communities = apply_louvain(G)
        girvan_newman_communities = apply_girvan_newman(G)

        louvain_result = display_community_results(G, "Louvain algorithm", louvain_communities)
        girvan_newman_result = display_community_results(G, "Girvan Newman algorithm", girvan_newman_communities)

        # Create a new Toplevel window for displaying results
        results_window = tk.Toplevel()
        results_window.title("Community Detection Results")

        # Create a Text widget to display Louvain algorithm results
        louvain_text = tk.Text(results_window, wrap="word", height=20, width=80)
        louvain_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Insert Louvain algorithm results into the Text widget
        louvain_text.insert("end", louvain_result)

        # Create a vertical scrollbar for Louvain algorithm results
        louvain_scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=louvain_text.yview)
        louvain_scrollbar.grid(row=0, column=1, sticky="ns")

        # Configure the Text widget to use the vertical scrollbar
        louvain_text.config(yscrollcommand=louvain_scrollbar.set)

        # Create a Text widget to display Girvan Newman algorithm results
        girvan_newman_text = tk.Text(results_window, wrap="word", height=20, width=80)
        girvan_newman_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Insert Girvan Newman algorithm results into the Text widget
        girvan_newman_text.insert("end", girvan_newman_result)

        # Create a vertical scrollbar for Girvan Newman algorithm results
        girvan_newman_scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=girvan_newman_text.yview)
        girvan_newman_scrollbar.grid(row=1, column=1, sticky="ns")

        # Configure the Text widget to use the vertical scrollbar
        girvan_newman_text.config(yscrollcommand=girvan_newman_scrollbar.set)

    else:
        coms=RB_HERE_for_directed_graph(G)
        RB_result = display_community_results(G, "RB algorithm", coms)
        girvan_newman_communities = apply_girvan_newman(G)
        girvan_newman_result = display_community_results(G, "Girvan Newman algorithm", girvan_newman_communities)
        results_window = tk.Toplevel()
        results_window.title("Community Detection Results")
        RB_text = tk.Text(results_window, wrap="word", height=20, width=80)
        RB_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Insert Louvain algorithm results into the Text widget
        RB_text.insert("end", RB_result)

        # Create a vertical scrollbar for Louvain algorithm results
        RB_scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=RB_text.yview)
        RB_scrollbar.grid(row=0, column=1, sticky="ns")

        # Configure the Text widget to use the vertical scrollbar
        RB_text.config(yscrollcommand=RB_scrollbar.set)



        girvan_newman_text = tk.Text(results_window, wrap="word", height=20, width=80)
        girvan_newman_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Insert Girvan Newman algorithm results into the Text widget
        girvan_newman_text.insert("end", girvan_newman_result)

        # Create a vertical scrollbar for Girvan Newman algorithm results
        girvan_newman_scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=girvan_newman_text.yview)
        girvan_newman_scrollbar.grid(row=1, column=1, sticky="ns")

        # Configure the Text widget to use the vertical scrollbar
        girvan_newman_text.config(yscrollcommand=girvan_newman_scrollbar.set)


Communities_frame = ttk.LabelFrame(root, text='Community')
Communities_frame.grid(row=5, column=0, sticky='ew')
tk.Label(Communities_frame, text='Community Results').grid(row=0, column=0)
results_button = ttk.Button(Communities_frame, text='Show Results', command=show_results)
results_button.grid(row=0, column=1)


def run_analysis(G, analysis_type):
    popup = tk.Toplevel()
    popup.title("Analysis Results")

    if analysis_type == 'Degree Distribution':
        degrees = []
        # Iterate over each node and its degree in the graph
        for node, degree in G.degree():
            degrees.append(degree)
        degree_count = Counter(degrees)
        total_nodes = float(G.number_of_nodes())
        prob = {}
        # Calculate the probability for each degree using a for-loop
        for degree, count in degree_count.items():
            prob[degree] = count / total_nodes

        deg, probabilities = zip(*sorted(prob.items()))


        # Generate the plot within the popup using a matplotlib figure
        fig = plt.figure(figsize=(10, 5))
        plt.bar(deg,probabilities, color='black')
        plt.title("Degree Distribution")
        plt.ylabel("Frequency")
        plt.xlabel("Degree")
        plt.close(fig)  # Close the figure to free up memory

        # Embed the figure in the popup window
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)
        canvas.draw()

    elif analysis_type == 'clustering coefficient':
        clustering = nx.clustering(G)

        # Extract the clustering values for plotting
        coeff_values = list(clustering.values())
        average_clustering = nx.average_clustering(G)
        # Plotting the histogram of clustering coefficients
        FIG = plt.figure(figsize=(8, 5))
        plt.hist(coeff_values, bins=10, color='green', alpha=0.7)
        plt.title('Clustering Coefficient Distribution\nAverage Clustering: {:.2f}'.format(average_clustering))
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.close(FIG)
        canvas = FigureCanvasTkAgg(FIG, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)
        canvas.draw()

    elif analysis_type == 'Average Path Length':

        if G.is_directed() and nx.is_strongly_connected(G)==False:
            message = "Graph is not connected; cannot compute average path length."
        else:
            apl = average_path_length(G)
            message = f"The average path length is: {apl:.2f}"



        tk.Label(popup, text=message).pack(padx=20, pady=20)

    elif analysis_type == 'percentage of each gender':
        genders = [data['gender'] for node, data in G.nodes(data=True)]
        gender_counts = Counter(genders)

        # Calculate percentages
        total_nodes = sum(gender_counts.values())
        gender_percentages = {gender: count / total_nodes * 100 for gender, count in gender_counts.items()}

        # Plotting
        labels = gender_percentages.keys()
        sizes = gender_percentages.values()

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Gender Distribution in Graph')
        plt.close(fig1)
        canvas = FigureCanvasTkAgg(fig1, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)
        canvas.draw()

    else:
        error_msg = "Unknown analysis type selected."
        tk.Label(popup, text=error_msg).pack(padx=20, pady=20)


def average_path_length(G):
    if G.is_directed():
        # Check if the graph is strongly connected
        if nx.is_strongly_connected(G):
            return nx.average_shortest_path_length(G)


    else:
        if nx.is_connected(G):
            return nx.average_shortest_path_length(G)


analysis_frame = ttk.LabelFrame(root, text='Analysis')
analysis_frame.grid(row=5, column=1, sticky='ew')  # Placed right under the layout frame

tk.Label(analysis_frame, text='Analysis Type').grid(row=0, column=0)
analysis_entry = ttk.Combobox(analysis_frame, values=(
    'Degree Distribution', 'clustering coefficient', 'Average Path Length', 'percentage of each gender'), state='readonly')
analysis_entry.grid(row=0, column=1)
analysis_entry.set('Degree Distribution')  # Default value

tk.Label(analysis_frame, text='Run Analysis').grid(row=1, column=0)
analysis_button = ttk.Button(analysis_frame, text='Analyze', command=lambda: run_analysis(G, analysis_entry.get()))
analysis_button.grid(row=1, column=1)


def plot_clustring(G, clustring_type):
    popup = tk.Toplevel()
    popup.title("clustring result")
    if clustring_type == 'Louvain algorithm':
        if G.is_directed():
            messagebox.showinfo("Clustering Error", "The Louvain algorithm is not suitable for directed graphs.")
            return

        partition = community_louvain.best_partition(G)
        # Create a color map for the communities
        num_communities = max(partition.values()) + 1
        print(num_communities)
        cmap = plt.get_cmap('viridis', num_communities)

        # Generate colors for each node based on its community using the colormap
        node_colors = []
        for node in G.nodes:
            color = cmap(partition[node])  # Get the color for this node based on its partition
            node_colors.append(color)

        # Draw the graph
        fig = plt.figure(figsize=(10, 5))
        pos = nx.spring_layout(G,scale=2)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos)
        plt.close(fig)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)
        canvas.draw()
    elif clustring_type == 'Girvan Newman':
        level3 = girvanstandard(G)
        num_communities = len(level3)
        cmap = plt.get_cmap('viridis', num_communities)
        # Map each node to its community
        community_map = {}
        #here we want to put number of the community beside the node itself to get the color
        for index, community in enumerate(level3):
            for node in community:
                community_map[node] = index

        node_colors = []
        for node in G.nodes:
            color = cmap(community_map[node])  # Get the color for this node based on its partition
            node_colors.append(color)


        fig = plt.figure(figsize=(10, 5))
        pos = nx.spring_layout(G,scale=2)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos)
        plt.close(fig)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)
        canvas.draw()
    elif clustring_type == 'partitioning by Class':
        communities = part_class(G)
        num_communities = len(communities)

        if num_communities == 0:
            return  # No communities to display

        # Creating one large figure with multiple subplots
        fig, axes = plt.subplots(1, num_communities, figsize=(num_communities * 7, 7))
        if num_communities == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        for ax, (classs, nodes) in zip(axes, communities.items()):
            subgraph = G.subgraph(nodes)
            ax.set_title(f"Class {classs}")
            pos = nx.spring_layout(subgraph,scale=2)  # Layout positions for nodes
            nx.draw(subgraph, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')
            ax.set_axis_off()  # Turn off the axis

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)

        plt.close(fig)

    elif clustring_type == 'partitioning by Gender':
        communities = part_gender(G)
        num_communities = len(communities)

        if num_communities == 0:
            return  # No communities to display

        # Creating one large figure with multiple subplots
        fig, axes = plt.subplots(1, num_communities, figsize=(num_communities * 7, 7))
        if num_communities == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        for ax, (classs, nodes) in zip(axes, communities.items()):
            subgraph = G.subgraph(nodes)
            ax.set_title(f"Class {classs}")
            pos = nx.spring_layout(subgraph,scale=2)  # Layout positions for nodes
            nx.draw(subgraph, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')
            ax.set_axis_off()  # Turn off the axis

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)

        plt.close(fig)
    elif clustring_type == 'RB algorithm(for directed)':
        coms=RB_HERE_for_directed_graph(G)

        communities = coms.communities

        num_communities = len(communities)

        # Create a colormap for coloring nodes based on communities
        cmap = plt.get_cmap('viridis', num_communities)

        # Map each node to its community
        community_map = {}
        for index, community in enumerate(communities):
            for node in community:
                community_map[node] = index

        # node colors based on community membership
        node_colors = [cmap(community_map[node]) for node in G.nodes()]

        # Create a figure and draw the network
        fig = plt.figure(figsize=(10, 5))
        pos = nx.spring_layout(G,scale=2)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos)
        plt.close(fig)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)
        canvas.draw()


    else:
        error_msg = "Unknown clustring type selected."
        tk.Label(popup, text=error_msg).pack(padx=20, pady=20)


clustring_frame = ttk.LabelFrame(root, text='clustring')
clustring_frame.grid(row=0, column=1, sticky='ew')  # Placed right under the layout frame

tk.Label(clustring_frame, text='clustring Type').grid(row=0, column=0)
clustring_entry = ttk.Combobox(clustring_frame, values=(
    'Girvan Newman', 'Louvain algorithm','RB algorithm(for directed)','partitioning by Class', "partitioning by Gender"), state='readonly')
clustring_entry.grid(row=0, column=1)
clustring_entry.set('Louvain algorithm')  # Default value

tk.Label(clustring_frame, text='plot clustring').grid(row=1, column=0)
clustring_button = ttk.Button(clustring_frame, text='clustring',
                              command=lambda: plot_clustring(G, clustring_entry.get()))
clustring_button.grid(row=1, column=1)


def evalutionofclustring(G, evaluationtype):
 selection = evaluationtype

 if(G.is_directed()==False):
    if selection == "Modularity":
        mod_louvain = calculatemodtolouvain(G)
        mod_newman = calculatemodtonewman(G)
        result_text.set(f"Louvain Modularity: {mod_louvain:.4f}\nNewman Modularity: {mod_newman:.10e}")
    elif selection == "NMI":
        nmi_louvain = Normalizemutualnformation_louvain(G)
        nmi_newman = Normalizemutualnformation_newman(G)
        result_text.set(f"Louvain NMI: {nmi_louvain:.4f}\nNewman NMI: {nmi_newman:.4f}")
    elif selection == "Conductance":
        cond_louvain = calculate_conductance(G)
        cond_newman = calculate_conductance_for_newman(G)
        louvain_results = "\n".join([f"Louvain Community {k}: {v:.4f}" for k, v in cond_louvain.items()])
        newman_results = "\n".join([f"Newman Community {k}: {v:.4f}" for k, v in cond_newman.items()])
        conductance_result = f"{louvain_results}\n\n\n{newman_results}"
        result_text.set(conductance_result)


 else:
     if selection == "Modularity":
         coms = RB_HERE_for_directed_graph(G)
         mod_newman = calculatemodtonewman(G)
         community_sets = [set(community) for community in coms.communities]
         modularity = RB_Modularity(G, community_sets)
         result_text.set(f"RB_pots Modularity: {modularity:.4f}\nNewman Modularity: {mod_newman:.10e}")
     elif selection == "NMI":
         NMI_RB= Normalizemutualnformation_RB(G)
         nmi_newman = Normalizemutualnformation_newman(G)
         result_text.set(f"RB_pots NMI: {NMI_RB:.4f}\nNewman NMI: {nmi_newman:.4f}")
     elif selection == "Conductance":
         Conductance_RB = calculate_conductance_for_RB(G)
         cond_newman = calculate_conductance_for_newman(G)
         newman_results = "\n".join([f"Newman Community {k}: {v:.4f}" for k, v in cond_newman.items()])
         rb_result = "\n".join([f"rb Community {k}: {v:.4f}" for k, v in Conductance_RB.items()])
         result_text.set(f"{rb_result}\n\n{newman_results}")




evaluation_frame = ttk.LabelFrame(root, text='evaluation')
evaluation_frame.grid(row=1, column=1, sticky='ew')  # Placed right under the layout frame

tk.Label(evaluation_frame, text='evaluation Type').grid(row=0, column=0)
evaluation_entry = ttk.Combobox(evaluation_frame, values=(
    "Modularity", "NMI", "Conductance"), state='readonly')
evaluation_entry.grid(row=0, column=1)
evaluation_entry.set('Modularity')  # Default value

result_text = tk.StringVar()
result_label = tk.Label(evaluation_frame, textvariable=result_text)
result_label.grid(row=2, columnspan=2)

evaluation_button = ttk.Button(evaluation_frame, text='Evaluate',
                               command=lambda: evalutionofclustring(G, evaluation_entry.get()))
evaluation_button.grid(row=1, column=1)


def show_results_in_popup(results):
    popup = tk.Toplevel()
    popup.title("Analysis Results")
    scroll_text = tk.Text(popup, wrap="word", height=20, width=80)
    scroll_text.pack(fill="both", expand=True)
    scroll_text.insert("end", results)


def show_pagerank_results():
    pagerank_scores = calculate_pagerank(G)
    top_n_nodes = get_top_n_nodes(pagerank_scores, n=5)
    result_text = "PageRank Scores:\n"
    for node, score in pagerank_scores.items():
        result_text += f"Node {node}: {score:.4f}"
        if node in top_n_nodes:
            result_text += " (Important)"
        result_text += "\n"
    show_results_in_popup(result_text)


def show_betweenness_results():
    betweenness_scores = calculate_betweenness_centrality(G)
    top_n_nodes = get_top_n_nodes(betweenness_scores, n=5)
    result_text = "Betweenness Centrality Scores:\n"
    for node, score in betweenness_scores.items():
        result_text += f"Node {node}: {score:.4f}"
        if node in top_n_nodes:
            result_text += " (Important)"
        result_text += "\n"
    show_results_in_popup(result_text)


def get_top_n_nodes(scores, n):
    sorted_nodes = sorted(scores, key=scores.get, reverse=True)
    return sorted_nodes[:n]


link_analysis_frame = ttk.LabelFrame(root, text='Link Analysis')
link_analysis_frame.grid(row=2, column=1, sticky='ew')
tk.Label(link_analysis_frame, text='PageRank').grid(row=0, column=0)
tk.Label(link_analysis_frame, text='Betweenness Centrality').grid(row=1, column=0)

pagerank_button = ttk.Button(link_analysis_frame, text='Analysis', command=show_pagerank_results)
pagerank_button.grid(row=0, column=1)

betweenness_button = ttk.Button(link_analysis_frame, text='Analysis', command=show_betweenness_results)
betweenness_button.grid(row=1, column=1)


def plot_filtering(G, filterType,value):
    popup = tk.Toplevel()
    popup.title("Filter result")
    try:
        value = float(value)
    except ValueError:
        # Handle the case where the value is not a valid float
        # Display an error message or perform other actions
        print("Invalid value entered")
        return
    if filterType == 'degree centrality':
        subgraph = filtergraphcentrality(G,value)
    elif filterType == 'closeness centrality':
        subgraph = filtergraphclosness(G,value)
    elif filterType == 'betweenness centrality':
        subgraph = filtergraphbetweenness(G,value)
    # Draw the graph
    fig = plt.figure(figsize=(10, 5))
    pos = nx.spring_layout(subgraph,scale=2)  # positions for all nodes
    nx.draw(subgraph, pos, with_labels=True)
    plt.close(fig)
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(expand=True, fill=tk.BOTH)
    canvas.draw()


# gui of filtring
filter_frame = ttk.LabelFrame(root, text='Filter Graph')
filter_frame.grid(row=3, column=1, sticky='ew')  # Placed right under the layout frame

tk.Label(filter_frame, text='Value').grid(row=0, column=0)
value_entry = ttk.Entry(filter_frame)
value_entry.grid(row=0, column=1)

tk.Label(filter_frame, text='options for filter').grid(row=0, column=0)
filter_entry = ttk.Combobox(filter_frame, values=(
    'degree centrality', 'closeness centrality', 'betweenness centrality'), state='readonly')
filter_entry.grid(row=1, column=1)
filter_entry.set('betweenness centrality')  # Default value

tk.Label(filter_frame, text='plot Graph after filtering').grid(row=1, column=0)
filter_button = ttk.Button(filter_frame, text='Filtering', command=lambda: plot_filtering(G, filter_entry.get(),value_entry.get()))
filter_button.grid(row=1, column=2)


def plot_filteringrange(G, filterType, range1, range2):
    popup = tk.Toplevel()
    popup.title("Filter result")

    range1 = float(range1)
    range2 = float(range2)

    if filterType == 'degree centrality':
        subgraph = filtergraphcentralityRange(G, range1, range2)
    elif filterType == 'closeness centrality':
        subgraph = filtergraphclosnessrange(G, range1, range2)
    elif filterType == 'betweenness centrality':
        subgraph = filtergraphbetweennessrange(G, range1, range2)
    # Draw the graph
    fig = plt.figure(figsize=(10, 5))
    pos = nx.spring_layout(subgraph,scale=2)  # positions for all nodes
    nx.draw(subgraph, pos, with_labels=True)
    plt.close(fig)
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(expand=True, fill=tk.BOTH)
    canvas.draw()


filterange_frame = ttk.LabelFrame(root, text='FilterRange Graph')
filterange_frame.grid(row=4, column=1, sticky='ew')  # Placed right under the layout frame

tk.Label(filterange_frame, text='range1').grid(row=0, column=0)
range_entry = ttk.Entry(filterange_frame)
range_entry.grid(row=0, column=1)

tk.Label(filterange_frame, text='range2').grid(row=1, column=0)
range1_entry = ttk.Entry(filterange_frame)
range1_entry.grid(row=1, column=1)

tk.Label(filterange_frame, text='options for filters').grid(row=2, column=0)
filterange_entry = ttk.Combobox(filterange_frame, values=(
    'degree centrality', 'closeness centrality', 'betweenness centrality'), state='readonly')
filterange_entry.grid(row=2, column=1)
filterange_entry.set('betweenness centrality')  # Default value

tk.Label(filterange_frame, text='plot Graph after filtering').grid(row=3, column=0)
filterange_button = ttk.Button(filterange_frame, text='Filtering',
                               command=lambda: plot_filteringrange(G, filterange_entry.get(), range_entry.get(),
                                                                   range1_entry.get()))
filterange_button.grid(row=3, column=1)


def visualize_communities(G):

    communities = part_class(G)

    column_index = 0  # Start placing plots at the first column
    for classs, nodes in communities.items():
        subgraph = G.subgraph(nodes)  # Create a subgraph with nodes of this class
        fig, ax = plt.subplots(figsize=(3, 3))  # Create a figure and add a subplot

        plt.title(f"Community of Class {classs}")
        pos = nx.spring_layout(subgraph)  # Layout for better visualization
        nx.draw(subgraph, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')

        # Display the figure in the Tkinter window using grid
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=column_index, column=0, sticky="nsew")  # Place each plot in a new column
        column_index += 1  # Increment column index for the next plot

        plt.close(fig)



root.mainloop()