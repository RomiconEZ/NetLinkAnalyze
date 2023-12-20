from sklearn import metrics, model_selection

from .Base import TemporalGraph
from .FormFeaturesDF import cnt_ft_wth_lbls_for_edges_of_st_gr


def train_test_split_temporal_graph(edge_list: list, split_ratio: float):
    """
    Dividing the sample into feature generation and prediction parts
    """
    edge_list_feature_build_part = edge_list[: int(len(edge_list) * split_ratio)]
    edge_list_prediction_part = edge_list[len(edge_list_feature_build_part) :]
    return edge_list_feature_build_part, edge_list_prediction_part


def get_performance(temporalG: TemporalGraph, split_ratio: float, cls_model, verbose: bool = False):
    """
    Evaluate the performance of a predictive model on a temporal graph.

    Args:
        temporalG (graphs.TemporalGraph): The temporal graph to be analyzed.
        split_ratio (float): The ratio for splitting the graph into training and testing data.
        cls_model: classification model for predicting the appearance of an edge.
    Returns:
        float: The AUC (Area Under the Curve) score of the model.

    The function calculates features for edges, labels them, splits the data into training and testing sets, and
    then trains a logistic regression model to predict the labels. It finally computes the AUC score.
    """

    X, y = cnt_ft_wth_lbls_for_edges_of_st_gr(temporalG, split_ratio, verbose)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)

    if verbose:
        print("Start of classification model training.")

    cls_model.fit(X_train, y_train)

    auc = metrics.roc_auc_score(y_true=y_test, y_score=cls_model.predict_proba(X_test)[:, 1])

    return auc
