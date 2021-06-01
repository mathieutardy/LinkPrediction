from graph_processor import GraphProcessor
from model_preprocessing import ModelPreprocessing
from model_training import ModelTraining
from graphsage import GraphSage
import argparse
from text2dict import ProcessAmazon



def main(args):

    # # Load metadata and save pickle
    # amazonProducts = ProcessAmazon(args.amazon_pickle)

    # Loading graph
    Processor = GraphProcessor(args.amazon_pickle)
    g = Processor.read_amazon_data()
    g = Processor.filter_comp(g)
    g = Processor.sub_sample_graph(g, 5)
    Processor.save_graph(g, args.path_graph)
    g = Processor.load_graph(args.path_graph)
    df = Processor.create_dataframe(g, int(1e4))

    # # Compute GraphSage Embeddings
    # GraphSageBuilder = GraphSage(args.path_node2vec_model,args.path_graph,args.path_sage_embedding)
    # GraphSageBuilder.train()

    # Preprocessing
    ModelPreprocessor = ModelPreprocessing(
        args.path_node2vec_model, args.path_sage_embedding, g
    )
    df = ModelPreprocessor.compute_cosine_similarity(
        df, ModelPreprocessor.embedding_dic_node2vec, "cosine_sim_node2vec"
    )
    df = ModelPreprocessor.compute_cosine_similarity(
        df, ModelPreprocessor.dic_embedding_sage, "cosine_sim_sage"
    )
    df = ModelPreprocessing.compute_resource_allocation_index(df, g)
    df = ModelPreprocessing.compute_adamic_adar(df, g)
    ModelPreprocessing.save_dataframe(df, args.path_comparison_dataset)

    # Training
    X_train, X_test, y_train, y_test = ModelTraining.prepare_data_for_training(
        df,
        [
            "cosine_sim_sage",
            "adamic_adar_index",
            "resource_allocation_index",
            "cosine_sim_node2vec",
        ],
    )  #
    results = ModelTraining.ensemble_classifier_training(
        X_train, X_test, y_train, y_test
    )
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link Prediction using Graphical Data")
    parser.add_argument(
        "--amazon_pickle", type=str, default="./data/amazonProducts.pkl"
    )
    parser.add_argument("--path_graph", type=str, default="./data/small_graph_bfs")
    parser.add_argument(
        "--path_node2vec_model", type=str, default="./data/node2vec.model"
    )
    parser.add_argument(
        "--path_sage_embedding", type=str, default="./data/SAGE_EMBEDDING"
    )
    parser.add_argument(
        "--path_comparison_dataset", type=str, default="./data/comparison_data.csv"
    )
    args = parser.parse_args()
    main(args)
