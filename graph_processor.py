import pickle
import networkx as nx
import numpy as np
import random

random.seed(0)
import pandas as pd


class GraphProcessor:
    def __init__(self, path):
        self.path = path

    def read_amazon_data(self):
        x = open(self.path, "rb")
        dic = pickle.load(x)

        edges = list()
        for prod_id in dic:
            prod = dic[prod_id]
            linked_prods = prod["CoPurchased"]
            if len(linked_prods) > 0:
                linked_prods = linked_prods.split(" ")
                for linked_id in linked_prods:
                    edges.append((prod_id, linked_id))

        g = nx.Graph()
        g.add_edges_from(edges)

        print("---- Initial Graph Size ----")
        print("Number of nodes: {}".format(g.number_of_nodes()))
        print("Number of edges: {}".format(g.number_of_edges()))

        return g

    def filter_comp(self, g):
        # filter on largest connected component
        largest_cc = max(nx.connected_components(g), key=len)
        bad_nodes = [node for node in g.nodes if node not in largest_cc]
        g.remove_nodes_from(bad_nodes)

        print("\n---- Keep largest connected components ----")
        print("Number of nodes: {}".format(g.number_of_nodes()))
        print("Number of edges: {}".format(g.number_of_edges()))

        return g

    def random_sample_nodes(self, g, sample_rate):
        # random sample about half of it
        all_nodes = list(g.nodes)
        nodes_to_keep = [
            all_nodes[random.randint(0, len(g.nodes))]
            for _ in range(int(sample_rate * len(g.nodes)))
        ]
        nodes_to_keep_without_dups = list(set(nodes_to_keep))

        # take largest component of that
        g = g.subgraph(nodes_to_keep_without_dups).copy()

        print("\n---- Random Selection of Nodes ----")
        print("Number of nodes: {}".format(g.number_of_nodes()))
        print("Number of edges: {}".format(g.number_of_edges()))

        return g

    def sub_sample_graph(self, g, depth_limit):
        """
        Use BFS to subsample graph
        """

        nodes_list = list(g.nodes)

        # Source node is the force node of nodes_list
        new_nodes = list(nx.bfs_tree(g, source=nodes_list[0], depth_limit=depth_limit))
        H = g.subgraph(new_nodes)

        print("\n---- BFS to sub sample graph ----")
        print("Number of nodes: {}".format(H.number_of_nodes()))
        print("Number of edges: {}".format(H.number_of_edges()))

        return H

    def save_graph(self, g, path):
        return nx.write_edgelist(g, path)

    def load_graph(self, path):
        g = nx.read_edgelist(path)
        print("\n--- Loading Graph ----")
        print("Number of nodes: {}".format(g.number_of_nodes()))
        print("Number of edges: {}".format(g.number_of_edges()))

        return g

    def random_combination(self, iterable, r):
        "Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), r))
        return list(pool[i] for i in indices)

    def create_dataframe(self, g, number_samples):

        """
        Create dataframe for node comparisons
        """

        print("\n---- Creating Dataframe ----")

        # Dataframe for possible samples
        linked_edges = list(g.edges())
        df = pd.DataFrame(linked_edges, columns=["node1", "node2"])
        df["link"] = 1

        # Random selection from possible combinations of two nodes
        nodes_list = list(g.nodes())
        nodes_columns = []
        for i in range(number_samples):
            nodes_columns.append(self.random_combination(nodes_list, 2))
        has_edge = [g.has_edge(nodes[0], nodes[1]) for nodes in nodes_columns]

        mapping = {
            True: 1,
            False: 0,
        }

        labels = [mapping[pred] for pred in has_edge]

        # Dataframe creation of random selection
        df2 = pd.DataFrame(nodes_columns, columns=["node1", "node2"])
        df2["link"] = labels
        df3 = pd.concat([df, df2])

        # Remove duplicates from random sample of negative examples
        df3.drop_duplicates()
        # Shuffle
        df3 = df3.sample(frac=1, random_state=1).reset_index(drop=True)

        return df3

    def save_dataframe(self, df, path):
        df.to_csv(path, index=False)
