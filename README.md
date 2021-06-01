# Network Science Project

This repository contains my network science project. We are building a recommendation system using an Amazon Metadata dataset [link](https://snap.stanford.edu/data/amazon-meta.html). We are trying to predict if two products are usually co-purchase d(usually bought together in the same basket). This relationship is represented by an edge in our graph. Our work is a link prediction problem.

Please find attached a report, a presentation and our code.

- main.py : You can run all of our project from there
- graph_processor.py: creates graph from dataset and convert it into reduced dataframe for comparisons
- model_preprocessing.py: computes metrics
- model_training.py: trains and predict results of classifiers.
- graphsage.py: trains a GraphSage and outputs an embedding for each node
- text2dict.py: takes the amazon dataset andconverts it into a dictionary which can be converted into a graph on networkx
