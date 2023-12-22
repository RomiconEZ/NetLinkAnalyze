import gc

import pandas as pd

from .Base import TemporalGraph
from .FeatureFormation import feature_for_absent_edges


def add_label_column(ft_bld_edge_df: pd.DataFrame, pred_adj_dict: dict[int, dict[int, [int]]]):
    """
    Add a label column to the DataFrame based on the presence of edges in the provided adjacency dictionary.

    Args:
        ft_bld_edge_df (pd.DataFrame): DataFrame containing features of edges.
        pred_adj_dict (dict): Adjacency dictionary to determine if an edge is present.

    The label is set to 1 if the edge (start_node, end_node) exists in the adjacency dictionary, else 0.
    """
    ft_bld_edge_df["label"] = ft_bld_edge_df.apply(
        lambda row: 1
        if (
            pred_adj_dict.get(row["start_node"]) is not None
            and pred_adj_dict[row["start_node"]].get(row["end_node"]) is not None
        )
        else 0,
        axis=1,
    )


def cnt_ft_wth_lbls_for_edges_of_st_gr(temporalG: TemporalGraph, split_ratio: float, verbose: bool = False):
    """
    Count features with labels for edges of the static graph derived from a temporal graph.

    Args:
        temporalG (TemporalGraph): The temporal graph to analyze.
        split_ratio (float): Ratio to split the graph for feature calculation.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Returns:
        tuple: Tuple containing features (X) and labels (y).
    """
    Edge_feature = feature_for_absent_edges(
        temporalG.get_static_graph(0, split_ratio).get_adjacency_dict_of_dicts(),
        temporalG.get_min_timestamp(),
        temporalG.get_max_timestamp(),
        verbose=verbose,
    )

    add_label_column(Edge_feature, temporalG.get_static_graph(split_ratio, 1).get_adjacency_dict_of_dicts())

    X = Edge_feature.drop(["label", "start_node", "end_node"], axis=1)
    y = Edge_feature["label"]

    del Edge_feature
    gc.collect()

    return X, y


def cnt_ft_for_edges_of_st_gr(temporalG: TemporalGraph, verbose: bool = False):
    """
    Count features for edges of the static graph derived from a temporal graph.

    Args:
        temporalG (TemporalGraph): The temporal graph to analyze.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Returns:
        DataFrame: DataFrame containing the features.
    """
    Edge_feature = feature_for_absent_edges(
        temporalG.get_static_graph(0, 1).get_adjacency_dict_of_dicts(),
        temporalG.get_min_timestamp(),
        temporalG.get_max_timestamp(),
        verbose=verbose,
    )

    X = Edge_feature.drop(["start_node", "end_node"], axis=1)
    del Edge_feature
    gc.collect()
    return X


def ft_for_edges_of_st_gr_frm_data_path(path_to_data, verbose: bool = False):
    """
    Generation of features for edges of a static graph using data from path_to_data.

    Args:
        path_to_data (str): Path to the data file for creating the temporal graph.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Returns:
        DataFrame: DataFrame containing the features of the static graph.
    """
    tmpGraph = TemporalGraph(path_to_data)

    X = cnt_ft_for_edges_of_st_gr(tmpGraph, verbose=verbose)

    return X
