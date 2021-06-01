"""
Input:
      - Read from graph data and filter on the small graph.
      - Read from node2vec embeddings.

Output:
      - GraphSage embeddings of the network. 
"""

import networkx as nx
import numpy as np
import pandas as pd
import pickle
from node2vec import Node2Vec
from gensim.models import Word2Vec

import dgl
from dgl.nn import SAGEConv
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp


class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_features):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, out_features, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# PRED part
class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            return graph.edata["score"]


# FULL model
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, neg_g, x):
        h = self.sage(g, x)
        return self.pred(g, h), self.pred(neg_g, h)


class GraphSage:
    def __init__(self, path_node2vec, path_graph, path_embeddings):
        self.path_node2vec = path_node2vec
        self.path_graph = path_graph
        self.path_embeddings = path_embeddings
        self.nodes_features, self.df = self.load_node2vec()
        self.G = self.load_graph()

    def load_node2vec(self):
        embedding_dic = {}
        model = Word2Vec.load(self.path_node2vec)

        # Retrieve node embeddings and corresponding subjects
        nodes_list = model.wv.index_to_key
        node_embeddings = model.wv.vectors

        for i, j in zip(nodes_list, node_embeddings):
            embedding_dic[i] = j

        df = pd.DataFrame.from_dict(embedding_dic, orient="index")
        nodes_features = torch.tensor(df.values.astype(float))

        return nodes_features, df

    def load_graph(self):
        g = nx.read_edgelist(self.path_graph)
        G = dgl.from_networkx(g)
        G.ndata["feat"] = self.nodes_features
        return G

    def construct_negative_graph(self, k):
        # Find all negative edges and split them for training and testing
        src, dst = self.G.edges()
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, self.G.num_nodes(), (len(src) * k,))
        return dgl.graph((neg_src, neg_dst), num_nodes=self.G.num_nodes())

    def prepare_data(self):
        # Split edge set for training and testing
        u, v = self.G.edges()
        eids = np.arange(self.G.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.2)
        train_size = self.G.number_of_edges() - test_size
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

        neg_graph = self.construct_negative_graph(k=1)
        neg_u, neg_v = neg_graph.edges()
        neg_eids = np.arange(neg_graph.number_of_edges())
        neg_eids = np.random.permutation(neg_eids)
        test_size = int(len(eids) * 0.15)
        train_size = neg_graph.number_of_edges() - test_size

        test_neg_u, test_neg_v = (
            neg_u[neg_eids[:test_size]],
            neg_v[neg_eids[:test_size]],
        )
        train_neg_u, train_neg_v = (
            neg_u[neg_eids[test_size:]],
            neg_v[neg_eids[test_size:]],
        )

        # separate negative and positive examples and build a graph on each
        train_pos_g = dgl.graph((train_pos_u, train_pos_v))
        train_neg_g = dgl.graph((train_neg_u, train_neg_v))

        test_pos_g = dgl.graph((test_pos_u, test_pos_v))
        test_neg_g = dgl.graph((test_neg_u, test_neg_v))

        return train_pos_g, train_neg_g, test_pos_g, test_neg_g

    def compute_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        scores = scores.view(len(scores))
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        return F.binary_cross_entropy_with_logits(scores, labels)

    def save_embeddings(self, model):
        node_embeddings = (
            model.sage(self.G, self.nodes_features.float()).detach().numpy()
        )
        new_embeds = pd.DataFrame({"node_ids": self.index})
        df_full = pd.concat([new_embeds, pd.DataFrame(node_embeddings)], axis=1)
        print("----- Saving Embeddings -----")
        df_full.to_csv(self.path_embeddings)

    def train(self):
        train_pos_g, train_neg_g, test_pos_g, test_neg_g = self.prepare_data()
        n_features = self.nodes_features.shape[1]
        model = Model(n_features, 100, 100)  # .to(device)
        opt = torch.optim.Adam(model.parameters())  # .to(device)
        for epoch in range(500):
            pos_score, neg_score = model(
                train_pos_g, train_neg_g, self.nodes_features.float()
            )
            loss = self.compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss.item())
            if epoch % 50 == 0:
                with torch.no_grad():
                    print(loss.item())
                    pos_score, neg_score = model(
                        test_pos_g, test_neg_g, self.nodes_features.float()
                    )
                    loss = self.compute_loss(pos_score, neg_score)
                    print(f"--- val epoch {epoch} --- {loss.item()}")

        self.save_embeddings(model)
