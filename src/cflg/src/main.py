import pandas as pd

from .CreateTables import graph_features_auc_score_tables_from_df
from .FormFeaturesDF import ft_for_edges_of_st_gr_frm_data_path


def graph_features_auc_score_tables(datasets_info: pd.DataFrame, cls_model=None, verbose=False):
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
    try:
        (
            latex_feature_network_table_1,
            latex_feature_network_table_2,
            latex_feature_network_table_3,
            latex_feature_network_table_4,
            latex_auc_table,
        ) = graph_features_auc_score_tables_from_df(datasets_info, cls_model=cls_model, verbose=verbose)
        return (
            latex_feature_network_table_1,
            latex_feature_network_table_2,
            latex_feature_network_table_3,
            latex_feature_network_table_4,
            latex_auc_table,
        )
    except Exception as e:
        print(e)
        return None, None, None, None, None


def features_for_edges_of_static_graph(path_to_data, verbose=False):
    """
    Generate features for edges of the static graph from data file

    Args:
        path_to_data: the path to the data file in the format: string - "num_node_1 num_node_2 timestamp".
            The data starts with the 3rd line. (the first two lines of the file are skipped)
    Returns:
        pandas.DataFrame: features for edges of the static graph
    """
    try:
        return ft_for_edges_of_st_gr_frm_data_path(path_to_data, verbose)
    except Exception as e:
        print(e)
        return None
