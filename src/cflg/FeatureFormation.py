import gc

import numpy as np
import pandas as pd
from numpy import ndarray


def common_neighbours(u: int, v: int, adjacency_dict_of_dicts: dict[int, dict[int, [int]]]):
    return len(list(adjacency_dict_of_dicts[u].keys() & adjacency_dict_of_dicts[v].keys()))


def adamic_adar(u: int, v: int, adjacency_dict_of_dicts: dict[int, dict[int, [int]]]):
    common_neigh = np.array(
        [
            len(adjacency_dict_of_dicts[k])
            for k in (adjacency_dict_of_dicts[u].keys() & adjacency_dict_of_dicts[v].keys())
        ]
    )
    return np.sum(1.0 / np.log(common_neigh))


def jaccard_coefficient(u: int, v: int, adjacency_dict_of_dicts: dict[int, dict[int, [int]]]):
    common_neigh_count = common_neighbours(u, v, adjacency_dict_of_dicts)
    if common_neigh_count == 0:
        return 0
    return common_neigh_count / len(list(adjacency_dict_of_dicts[u].keys() | adjacency_dict_of_dicts[v].keys()))


def preferential_attachment(u: int, v: int, adjacency_dict_of_dicts: dict[int, dict[int, [int]]]):
    return len(adjacency_dict_of_dicts[u]) * len(adjacency_dict_of_dicts[v])


# 3 functions for calculating weights
# An np array of timestamps of all edges can be passed to all functions as t
def temporal_weighting_w_linear(t: np.array, t_min: int, t_max: int, l: float = 0.2) -> float:
    return l + (1 - l) * (t - t_min) / (t_max - t_min)  # w_linear


def temporal_weighting_w_exponential(t: np.array, t_min: float, t_max: float, l: float = 0.2) -> float:
    return l + (1 - l) * (np.exp(3 * (t - t_min) / (t_max - t_min)) - 1) / (np.exp(3) - 1)  # w_exponential


def temporal_weighting_w_square_root(t: np.array, t_min: float, t_max: float, l: float = 0.2) -> float:
    return l + (1 - l) * np.sqrt((t - t_min) / (t_max - t_min))  # w_square_root


# 7 functions for aggregation of weights of edges adjacent to a node


def aggregation_of_node_activity_quantile(
    edges_weights: list, quantile: float
) -> list[int | float | complex | ndarray]:
    # zero - 0
    # first - 0.25
    # second - 0.5
    # third - 0.75
    # fourth - 1
    return [np.quantile(weights_set, quantile) for weights_set in edges_weights]


def aggregation_of_node_activity_sum(edges_weights: list) -> list[ndarray]:
    return [np.sum(weights_set, 0) for weights_set in edges_weights]


def aggregation_of_node_activity_mean(edges_weights: list) -> list[ndarray]:
    return [np.mean(weights_set, 0) for weights_set in edges_weights]


# 4 functions for combining statistics on pairs of nodes


def combining_node_activity_sum(aggregarion_node_one: np.array, aggregation_node_two: np.array) -> np.array:
    return aggregarion_node_one + aggregation_node_two


def combining_node_activity_absolute_diference(
    aggregarion_node_one: np.array, aggregation_node_two: np.array
) -> np.array:
    return np.abs(aggregarion_node_one - aggregation_node_two)


def combining_node_activity_minimum(aggregarion_node_one: np.array, aggregation_node_two: np.array) -> np.array:
    return np.min(
        np.concatenate([aggregarion_node_one[:, np.newaxis], aggregation_node_two[:, np.newaxis]], axis=1), axis=1
    )


def combining_node_activity_maximum(aggregarion_node_one: np.array, aggregation_node_two: np.array) -> np.array:
    return np.max(
        np.concatenate([aggregarion_node_one[:, np.newaxis], aggregation_node_two[:, np.newaxis]], axis=1), axis=1
    )


def temporal_weighting(edge: pd.DataFrame, t_min: int, t_max: int):
    """
    Time weighting: calculation of three weights for each edge based on their timestamps.
    """
    edge["wl"] = temporal_weighting_w_linear(edge["timestamp"], t_min, t_max)
    edge["we"] = temporal_weighting_w_exponential(edge["timestamp"], t_min, t_max)
    edge["wsr"] = temporal_weighting_w_square_root(edge["timestamp"], t_min, t_max)


def replace_nan(df, columns, start_node_column, end_node_column):
    """
    Replacing NaN with [] in numeric cells and matching numbers for vertices
    that were only in either end_node or start_node.
    """
    for column in columns:
        df[column] = df[column].apply(lambda x: [] if not isinstance(x, list) else x)

    df[start_node_column] = np.where(np.isnan(df[start_node_column]), df[end_node_column], df[start_node_column])

    return df


def aggregation_of_node_activity(edges_weights_for_node: pd.DataFrame):
    """
    Aggregation of node activity based on 7 functions:
    zero, first, second, third, fourth quantiles; sum and weight average of edges adjacent to the vertex
    """
    names_of_weights = ["wl", "we", "wsr"]
    number_of_quantile_dict = {"zeroth": 0, "first": 0.25, "second": 0.5, "third": 0.75, "fourth": 1}
    for name_wgh in names_of_weights:
        for key_nm_qnt, value_nm_qnt in number_of_quantile_dict.items():
            column_name = f"node_activity_{key_nm_qnt}_quantile_{name_wgh}"
            edges_weights_for_node[column_name] = aggregation_of_node_activity_quantile(
                edges_weights_for_node[name_wgh], value_nm_qnt
            )
        column_name = f"node_activity_sum_{name_wgh}"
        edges_weights_for_node[column_name] = aggregation_of_node_activity_sum(edges_weights_for_node[name_wgh])
        column_name = f"node_activity_mean_{name_wgh}"
        edges_weights_for_node[column_name] = aggregation_of_node_activity_mean(edges_weights_for_node[name_wgh])
        edges_weights_for_node = edges_weights_for_node.drop([name_wgh], axis=1)

    edges_weights_for_node.rename(columns={"start_node": "node"}, inplace=True)

    return edges_weights_for_node


def make_edges_weights_adjacent_to_node(edge: pd.DataFrame):
    """
    Forming a dataframe with weights of edges adjacent to the vertex.
    The string corresponds to a certain vertex.
    """
    grouped_by_start_node = (
        edge.groupby("start_node")
        .agg({"wl": lambda x: list(x), "we": lambda x: list(x), "wsr": lambda x: list(x)})
        .reset_index()
    )

    grouped_by_end_node = (
        edge.groupby("end_node")
        .agg({"wl": lambda x: list(x), "we": lambda x: list(x), "wsr": lambda x: list(x)})
        .reset_index()
    )

    edges_weights_for_node = pd.merge(
        grouped_by_start_node, grouped_by_end_node, left_on="start_node", right_on="end_node", how="outer"
    )
    edges_weights_for_node = replace_nan(
        edges_weights_for_node, ["wl_x", "we_x", "wsr_x", "wl_y", "we_y", "wsr_y"], "start_node", "end_node"
    )

    edges_weights_for_node["wl"] = edges_weights_for_node["wl_x"] + edges_weights_for_node["wl_y"]
    edges_weights_for_node["we"] = edges_weights_for_node["we_x"] + edges_weights_for_node["we_y"]
    edges_weights_for_node["wsr"] = edges_weights_for_node["wsr_x"] + edges_weights_for_node["wsr_y"]

    edges_weights_for_node = edges_weights_for_node.drop(
        ["end_node", "wl_x", "we_x", "wsr_x", "wl_y", "we_y", "wsr_y"], axis=1
    )
    edges_weights_for_node["start_node"] = edges_weights_for_node["start_node"].astype(int)
    edges_weights_for_node[["wl", "we", "wsr"]] = edges_weights_for_node[["wl", "we", "wsr"]].apply(
        lambda x: np.array(x)
    )
    edges_weights_for_node = edges_weights_for_node.sort_values(by="start_node")

    return edges_weights_for_node


def split_list_cell(df: pd.DataFrame, column_name: str):
    """
    Splitting the list into separate columns with automatic name generation.
    """
    # Generating column names
    new_columns = [str(i) for i in range(len(df[column_name].iloc[0]))]

    df[new_columns] = df[column_name].apply(pd.Series)

    return df.drop(column_name, axis=1)


def count_static_topological_features(df: pd.DataFrame, adjacency_dict_of_dicts: dict[int, dict[int, [int]]]):
    """
    Calculation of static topological features.
    """
    df["common_neighbours"] = df.apply(
        lambda row: common_neighbours(row["start_node"], row["end_node"], adjacency_dict_of_dicts), axis=1
    )
    df["adamic_adar"] = df.apply(
        lambda row: adamic_adar(row["start_node"], row["end_node"], adjacency_dict_of_dicts), axis=1
    )
    df["jaccard_coefficient"] = df.apply(
        lambda row: jaccard_coefficient(row["start_node"], row["end_node"], adjacency_dict_of_dicts), axis=1
    )
    df["preferential_attachment"] = df.apply(
        lambda row: preferential_attachment(row["start_node"], row["end_node"], adjacency_dict_of_dicts), axis=1
    )


def union_column_to_one(df, union_columns_name, new_one_column_name):
    df[new_one_column_name] = df[union_columns_name].apply(lambda row: np.hstack(row.values), axis=1)


def combining_node_activity_for_absent_edge(node_feature_df: pd.DataFrame, edge_feature_df: pd.DataFrame):
    """
    Combine node activity features for non-existent edges in a graph.

    This function creates new features for each edge by combining the features of the start and end nodes.
    The combination is based on various aggregation functions and quantiles.

    Args:
        node_feature_df (pd.DataFrame): DataFrame containing features of nodes.
        edge_feature_df (pd.DataFrame): DataFrame containing features of edges.

    """
    prefixes = ["wl", "we", "wsr"]
    quantiles = ["zeroth", "first", "second", "third", "fourth"]
    aggregations = ["sum", "mean"]

    # Generating column names for node features
    node_feature_columns_names = [
        f"node_activity_{quantile}_quantile_{prefix}" for prefix in prefixes for quantile in quantiles
    ] + [f"node_activity_{aggregation}_{prefix}" for prefix in prefixes for aggregation in aggregations]

    # Combining columns into one
    union_column_name = "node_feature"
    union_column_to_one(node_feature_df, node_feature_columns_names, union_column_name)
    node_feature_df = node_feature_df.drop(node_feature_columns_names, axis=1)

    # Creating a dictionary for faster access
    node_feature_dict = node_feature_df.set_index("node")[union_column_name].to_dict()

    del node_feature_df
    gc.collect()

    num_of_feature = len(node_feature_columns_names)
    num_of_ag_func = 4

    # Forming column names for edge features
    edge_feature_column_names = [
        [str(i * num_of_feature + j) for j in range(1, num_of_feature + 1)] for i in range(num_of_ag_func)
    ]

    # Collecting node features for each edge into a 2x(number of edges) array
    node_ids = edge_feature_df[["start_node", "end_node"]].values
    start_node_features = np.array([node_feature_dict[node_id] for node_id in node_ids[:, 0]])
    end_node_features = np.array([node_feature_dict[node_id] for node_id in node_ids[:, 1]])

    # Calculating features for edges
    fun_names = [
        combining_node_activity_sum,
        combining_node_activity_absolute_diference,
        combining_node_activity_minimum,
        combining_node_activity_maximum,
    ]
    for index, comb_func in enumerate(fun_names):
        edge_feature_df[edge_feature_column_names[index]] = comb_func(start_node_features, end_node_features)


def form_start_end_node_df(adjacency_dict_of_dicts: dict[int, dict[int, list[int]]]) -> pd.DataFrame:
    """
    Form a DataFrame of start and end nodes for edges that are not present in the adjacency dictionary.

    Args:
        adjacency_dict_of_dicts (dict): A dictionary of dictionaries representing the adjacency list of the graph.

    Returns:
        pd.DataFrame: A DataFrame with columns 'start_node' and 'end_node' for missing edges.
    """
    all_edge_number_set = set(adjacency_dict_of_dicts.keys())

    rows = []
    for start_node in all_edge_number_set:
        end_nodes = adjacency_dict_of_dicts[start_node]
        missing_end_nodes = all_edge_number_set - set(end_nodes.keys())

        rows.extend((start_node, end_node) for end_node in missing_end_nodes if start_node < end_node)

    start_end_node_df = pd.DataFrame(rows, columns=["start_node", "end_node"])
    return start_end_node_df


def form_unique_node_set(df: pd.DataFrame, columns_name: list[str]) -> set:
    """
    Form a set of unique nodes from specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing node data.
        columns_name (list[str]): List of column names to extract unique nodes from.

    Returns:
        set: A set of unique nodes.
    """
    unique_node_set = set()

    for column_name in columns_name:
        unique_node_set.update(df[column_name].unique())

    return unique_node_set


def form_edge_set_for_node_with_feature(
    adjacency_dict_of_dicts: dict[int, dict[int, list[int]]], unique_node_set_for_count_node_act: set
) -> pd.DataFrame:
    """
    Form a DataFrame of edges for nodes with specified features.

    Args:
        adjacency_dict_of_dicts (dict): A dictionary of dictionaries representing the adjacency list of the graph.
        unique_node_set_for_count_node_act (set): A set of nodes to consider for forming the edge set.

    Returns:
        pd.DataFrame: A DataFrame with columns 'start_node', 'end_node', and 'timestamp' for the edges.
    """
    rows = []

    for start_node in unique_node_set_for_count_node_act:
        end_nodes = set(adjacency_dict_of_dicts.get(start_node, {}).keys())
        common_nodes = unique_node_set_for_count_node_act & end_nodes

        for end_node in common_nodes:
            timestamps = adjacency_dict_of_dicts[start_node][end_node]
            rows.extend((start_node, end_node, timestamp) for timestamp in timestamps)

    edge_set_for_node_with_feature = pd.DataFrame(rows, columns=["start_node", "end_node", "timestamp"])
    return edge_set_for_node_with_feature


def feature_for_absent_edges(
    adjacency_dict_of_dicts: dict[int, dict[int, [int]]], t_min: int, t_max: int, verbose: bool = False
):
    """
    Generate a DataFrame with features for edges that are absent in the graph.
    """
    # DataFrame of edges that are absent in the graph and for which features need to be calculated
    edge_feature_df = form_start_end_node_df(adjacency_dict_of_dicts)
    if verbose:
        print("Got the edges for which we need to calculate the features")
    # All unique vertices that are incident to edges in edge_feature_df
    unique_node_set_for_count_node_act = form_unique_node_set(edge_feature_df, ["start_node", "end_node"])

    # Select edges that are only connected to vertices for which features need to be calculated
    edge = form_edge_set_for_node_with_feature(adjacency_dict_of_dicts, unique_node_set_for_count_node_act)

    # Weighting of edges
    temporal_weighting(edge, t_min, t_max)
    if verbose:
        print("Weighted the edges")
    # DataFrame of vertices and weights of edges incident to them
    edges_weights_adjacent_to_node = make_edges_weights_adjacent_to_node(edge)

    # Get DataFrame of node activities: |node|21 x node activity|
    node_feature_df = aggregation_of_node_activity(edges_weights_adjacent_to_node)
    if verbose:
        print("Aggregated weights")
    # высчитываем признаки для ребер
    combining_node_activity_for_absent_edge(node_feature_df, edge_feature_df)
    if verbose:
        print("Calculated 84 features for the edge")
    count_static_topological_features(edge_feature_df, adjacency_dict_of_dicts)
    if verbose:
        print("Calculated static topological features")
    return edge_feature_df
