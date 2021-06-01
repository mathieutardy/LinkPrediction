from tqdm import tqdm
from itertools import combinations
from time import time
from node2vec import Node2Vec
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import networkx as nx


class ModelPreprocessing:
    def __init__(self, save_path, sage_embedding_path, g):
        self.save_path = save_path
        self.sage_embedding_path = sage_embedding_path
        self.embedding_dic_node2vec = {}
        self.load_node2vec_embedding()
        self.dic_embedding_sage = {}
        self.load_sage_embedding(sage_embedding_path)
        self.g = g

    def train_node2vec(self):
        # Train Node2Vec

        # Generate walks - takes around 2 minutes
        node2vec = Node2Vec(
            self.train_node2vecg, dimensions=64, walk_length=8, num_walks=100, workers=1
        )  # temp_folder= path + "temp/"

        # Fitting node2vec - takes around 5 minutes
        t = time()
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        minutes = (time() - t) / 60
        print("Node2Vec training took {:.2f} minutes".format(minutes))

        # Save Embeddings
        # model.wv.save_word2vec_format(self.save_path)
        model.save(self.save_path)

    def load_node2vec_embedding(self):
        try:
            model = Word2Vec.load(self.save_path)
        except Exception:
            self.train_node2vec()

        # Retrieve node embeddings and corresponding subjects
        nodes_list = model.wv.index_to_key
        node_embeddings = model.wv.vectors

        for i, j in zip(nodes_list, node_embeddings):
            self.embedding_dic_node2vec[i] = j

    def load_sage_embedding(self, path):

        # Open Sage Embedding
        df_sage = pd.read_csv(path)
        df_sage.drop(["Unnamed: 0"], axis=1)

        # Create dictionnary
        nodes = df_sage.iloc[:, 1]
        embeddings = list(df_sage.iloc[:, 2:].to_numpy())
        self.dic_embedding_sage = dict(zip(nodes, embeddings))

    def compute_cosine_similarity(self, df, embedding, metric_name):
        """
        Use an embedding to compute cosine similarity score
        between two nodes.
        Add the value to a dataframe and return the dataframe.
        """
        pairs = df[["node1", "node2"]].to_numpy()
        embeddings_nodes1 = np.array([embedding[key] for key in pairs[:, 0]])
        embeddings_nodes2 = np.array([embedding[key] for key in pairs[:, 1]])
        scalar_product = np.sum(embeddings_nodes1 * embeddings_nodes2, axis=1)
        norms = np.linalg.norm(embeddings_nodes1, axis=1) * np.linalg.norm(
            embeddings_nodes2, axis=1
        )
        cosine_similarity = scalar_product / norms
        df[metric_name] = cosine_similarity

        return df

    @staticmethod
    def compute_resource_allocation_index(df, g):
        nodes_list = df[["node1", "node2"]].to_numpy()
        list_tuples = list(nx.resource_allocation_index(g, nodes_list))
        ra = [values[2] for values in list_tuples]
        df["resource_allocation_index"] = ra

        return df

    @staticmethod
    def compute_adamic_adar(df, g):
        nodes_list = df[["node1", "node2"]].to_numpy()
        list_tuples = list(nx.adamic_adar_index(g, nodes_list))
        aa = [values[2] for values in list_tuples]
        df["adamic_adar_index"] = aa

        return df

    @staticmethod
    def save_dataframe(df, path):
        df.to_csv(path, index=False)
