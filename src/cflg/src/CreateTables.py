import random

import pandas as pd

from .Base import SelectApproach, TemporalGraph
from .ModelPerformance import get_performance

network_rus_name = "Сеть"
cat_graph_rus_name = "Категория"
nodes_rus_name = "Вершины"
type_of_edge_rus_name = "Тип ребер"
edges_rus_name = "Ребра"
dens_rus_name = "Плот."
part_of_nodes_rus_name = "Доля вершин"
scc_rus_name = "КСС"
num_of_nodes_in_big_scc_rus_name = "Вершины в наиб.КСС"
num_of_edges_in_big_scc_rus_name = "Ребра в наиб.КСС"
radius_sb_rus_name = "Радиус(ск)"
diameter_sb_rus_name = "Диаметр(ск)"
procentile_sb_rus_name = "90проц.расст.(ск)"
radius_rnc_rus_name = "Радиус(свв)"
diameter_rnc_rus_name = "Диаметр(свв)"
procentile_rnc_rus_name = "90проц.расст.(свв)"
asst_fact_rus_name = "Коэф.ассорт."
avr_cl_fact_rus_name = "Ср.кл.коэф."
auc_rus_name = "AUC"


def get_stats(network_info, cls_model=None, verbose=False):
    """
    Generate statistics for a given network.

    Args:
        network_info (dict): Information about the network including path and other metadata.
        cls_model: classification model for predicting the appearance of an edge.
    Returns:
        dict: A dictionary containing various statistics and metrics of the network.
    """
    tmpGraph = TemporalGraph(network_info["Path"])
    staticGraph = tmpGraph.get_static_graph(0.0, 1.0)

    adjacency_dict_of_dicts = staticGraph.get_adjacency_dict_of_dicts()
    node1 = random.choice(list(adjacency_dict_of_dicts.keys()))
    node2 = random.choice(list(adjacency_dict_of_dicts[node1].keys()))

    snowball_sample_approach = SelectApproach(node1, node2)
    random_selected_vertices_approach = SelectApproach()
    sg_sb = snowball_sample_approach(staticGraph.get_largest_connected_component())
    sg_rsv = random_selected_vertices_approach(staticGraph.get_largest_connected_component())

    result = {}
    try:
        result[network_rus_name] = network_info["Label"]
    except KeyError:
        result[network_rus_name] = None

    try:
        result[cat_graph_rus_name] = network_info["Category"]
    except KeyError:
        result[cat_graph_rus_name] = None

    try:
        result[nodes_rus_name] = staticGraph.count_vertices()
    except Exception:
        result[nodes_rus_name] = None

    try:
        result[type_of_edge_rus_name] = network_info["Edge type"]
    except KeyError:
        result[type_of_edge_rus_name] = None

    try:
        result[edges_rus_name] = staticGraph.count_edges()
    except Exception:
        result[edges_rus_name] = None

    try:
        result[dens_rus_name] = staticGraph.density()
    except Exception:
        result[dens_rus_name] = None

    try:
        result[part_of_nodes_rus_name] = staticGraph.share_of_vertices()
    except Exception:
        result[part_of_nodes_rus_name] = None

    try:
        result[scc_rus_name] = staticGraph.get_number_of_connected_components()
        if verbose:
            print("Retrieved the number of connected components")
    except Exception:
        result[scc_rus_name] = None

    try:
        result[num_of_nodes_in_big_scc_rus_name] = staticGraph.get_largest_connected_component().count_vertices()
        if verbose:
            print("Retrieved the number of vertices in the largest connected component")
    except Exception:
        result[num_of_nodes_in_big_scc_rus_name] = None

    try:
        result[num_of_edges_in_big_scc_rus_name] = staticGraph.get_largest_connected_component().count_edges()
    except Exception:
        result[num_of_edges_in_big_scc_rus_name] = None

    try:
        result[radius_sb_rus_name] = staticGraph.get_radius(sg_sb)
    except Exception:
        result[radius_sb_rus_name] = None

    try:
        result[diameter_sb_rus_name] = staticGraph.get_diameter(sg_sb)
    except Exception:
        result[diameter_sb_rus_name] = None

    try:
        result[procentile_sb_rus_name] = staticGraph.percentile_distance(sg_sb)
    except Exception:
        result[procentile_sb_rus_name] = None

    try:
        result[radius_rnc_rus_name] = staticGraph.get_radius(sg_rsv)
    except Exception:
        result[radius_rnc_rus_name] = None

    try:
        result[diameter_rnc_rus_name] = staticGraph.get_diameter(sg_rsv)
    except Exception:
        result[diameter_rnc_rus_name] = None

    try:
        result[procentile_rnc_rus_name] = staticGraph.percentile_distance(sg_rsv)
    except Exception:
        result[procentile_rnc_rus_name] = None

    try:
        result[asst_fact_rus_name] = staticGraph.assortative_factor()
        if verbose:
            print("Retrieved the assortative coefficient")
    except Exception:
        result[asst_fact_rus_name] = None

    try:
        result[avr_cl_fact_rus_name] = staticGraph.average_cluster_factor()
        if verbose:
            print("Retrieved the average clustering coefficient")
    except Exception:
        result[avr_cl_fact_rus_name] = None
    try:
        if cls_model is not None:
            result[auc_rus_name] = get_performance(tmpGraph, 0.67, cls_model, verbose=verbose)
        else:
            result[auc_rus_name] = None
    except Exception:
        result[auc_rus_name] = None

    return result


def graph_features_auc_score_tables_from_df(datasets_info: pd.DataFrame, cls_model=None, verbose=False):
    """
    Generate LaTeX tables of network features from a DataFrame of datasets information.

    Args:
        datasets_info (pd.DataFrame): DataFrame with columns: 'Network', 'Label', 'Category', 'Edge type', 'Path'.
            Path - the path to the data file in the format: string - "num_node_1 num_node_2 timestamp".
            The data starts with the 3rd line. (the first two lines of the file are skipped)
        cls_model: classification model for predicting the appearance of an edge.
    Returns:
        tuple: A tuple of LaTeX strings for different feature tables of the networks.
    """
    # Creating a DataFrame of network statistics
    table = pd.DataFrame(
        [
            get_stats(dict(network_info), cls_model=cls_model, verbose=verbose)
            for index, network_info in datasets_info.iterrows()
        ]
    )
    # Specifying columns for different feature tables
    columns_to_include_to_feature_network_table_1 = [
        network_rus_name,
        cat_graph_rus_name,
        nodes_rus_name,
        type_of_edge_rus_name,
        edges_rus_name,
        dens_rus_name,
        part_of_nodes_rus_name,
    ]
    columns_to_include_to_feature_network_table_2 = [
        network_rus_name,
        scc_rus_name,
        num_of_nodes_in_big_scc_rus_name,
        num_of_edges_in_big_scc_rus_name,
    ]
    columns_to_include_to_feature_network_table_3 = [
        network_rus_name,
        radius_sb_rus_name,
        diameter_sb_rus_name,
        procentile_sb_rus_name,
        radius_rnc_rus_name,
        diameter_rnc_rus_name,
        procentile_rnc_rus_name,
    ]
    columns_to_include_to_feature_network_table_4 = [
        network_rus_name,
        asst_fact_rus_name,
        avr_cl_fact_rus_name,
    ]

    columns_to_include_to_auc_table = [
        network_rus_name,
        auc_rus_name,
    ]
    # Generating LaTeX tables for network features
    latex_feature_network_table_1 = table.to_latex(
        formatters={
            nodes_rus_name: lambda x: f"{x:,}",
            edges_rus_name: lambda x: f"{x:,}",
            dens_rus_name: lambda x: f"{x:.6f}",
            part_of_nodes_rus_name: lambda x: f"{x:.6f}",
        },
        column_format=r"l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c",
        index=False,
        caption=("Features for networks"),
        label="Table: Features for networks",
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table_1,
    )
    latex_feature_network_table_2 = table.to_latex(
        formatters={
            scc_rus_name: lambda x: f"{x:,}",
            num_of_nodes_in_big_scc_rus_name: lambda x: f"{x:,}",
            num_of_edges_in_big_scc_rus_name: lambda x: f"{x:,}",
        },
        column_format=r"l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c",
        index=False,
        caption=("Features for networks"),
        label="Table: Features for networks",
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table_2,
    )

    latex_feature_network_table_3 = table.to_latex(
        formatters={
            radius_sb_rus_name: lambda x: f"{x:.2f}",
            diameter_sb_rus_name: lambda x: f"{x:.2f}",
            procentile_sb_rus_name: lambda x: f"{x:.2f}",
            radius_rnc_rus_name: lambda x: f"{x:.2f}",
            diameter_rnc_rus_name: lambda x: f"{x:.2f}",
            procentile_rnc_rus_name: lambda x: f"{x:.2f}",
        },
        column_format=r"l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c",
        index=False,
        caption=("Features for networks"),
        label="Table: Features for networks",
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table_3,
    )
    latex_feature_network_table_4 = table.to_latex(
        formatters={
            asst_fact_rus_name: lambda x: f"{x:.2f}",
            avr_cl_fact_rus_name: lambda x: f"{x:.2f}",
        },
        column_format=r"l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c",
        index=False,
        caption=("Features for networks"),
        label="Table: Features for networks",
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_feature_network_table_4,
    )
    latex_auc_table = table.to_latex(
        formatters={
            auc_rus_name: lambda x: f"{x:.2f}",
        },
        column_format=r"l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c",
        index=False,
        caption=("Prediction accuracy of edges"),
        label="Table: AUC",
        escape=False,
        multicolumn=False,
        columns=columns_to_include_to_auc_table,
    )
    return (
        latex_feature_network_table_1,
        latex_feature_network_table_2,
        latex_feature_network_table_3,
        latex_feature_network_table_4,
        latex_auc_table,
    )
