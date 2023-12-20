from pathlib import Path

import pandas as pd
from IPython.core.display_functions import display
from sklearn import linear_model, pipeline, preprocessing

from src.cflg import features_for_edges_of_static_graph, graph_features_auc_score_tables


def test_graph_features_auc_score_tables() -> None:
    cls_model = pipeline.make_pipeline(
        preprocessing.StandardScaler(), linear_model.LogisticRegression(max_iter=10000, n_jobs=-1, random_state=42)
    )

    Networks = ["email-Eu-core-temporal-Dept3"]

    current_path = Path(__file__).parent
    networks_files_names = [str(current_path / name / f"out.{name}") for name in Networks]

    datasets_info = {
        "Network": Networks,
        "Label": ["EU"],
        "Category": ["Social"],
        "Edge type": ["Multi"],
        "Path": networks_files_names,
    }

    datasets_info = pd.DataFrame(datasets_info)
    datasets_info = datasets_info.iloc[0:1]
    print(datasets_info)
    print("---------------------------------------------")
    (
        latex_feature_network_table_1,
        latex_feature_network_table_2,
        latex_feature_network_table_3,
        latex_feature_network_table_4,
        latex_auc_table,
    ) = graph_features_auc_score_tables(datasets_info, cls_model=cls_model, verbose=True)
    print("---------------------------------------------")
    print(latex_feature_network_table_1)
    print("---------------------------------------------")
    print(latex_feature_network_table_2)
    print("---------------------------------------------")
    print(latex_feature_network_table_3)
    print("---------------------------------------------")
    print(latex_feature_network_table_4)
    print("---------------------------------------------")
    print(latex_auc_table)
    print("---------------------------------------------")
    return


def test_features_for_static_graph() -> None:
    def display_dataframe(df):
        with pd.option_context("display.max_columns", None):  # Показать все колонки
            display(df.head(5))  # Вывести первые 5 строк

    current_path = Path(__file__).parent
    name = "email-Eu-core-temporal-Dept3"
    path_to_data = str(current_path / name / f"out.{name}")

    X = features_for_edges_of_static_graph(path_to_data, verbose=True)

    display_dataframe(X)

    return
