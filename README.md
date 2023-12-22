# NetLinkAnalyze
## Описание:
### RU:
Аннотация: Программа «NetLinkAnalyze» предназначена для вычисления признакового описания связей в крупномасштабных сетях для дальнейшего обучения классификатора и предсказания появления новых связей в последующие моменты времени, а также для расчёта характеристик статических графов, включающий количество вершин и ребер, плотность, компоненты слабой связности, радиус, диаметр сети и коэффициент ассортативности. Обеспечивает получение результатов в удобном для анализа формате - LaTeX таблицы с эффективным использованием памяти вычислительного устройства для работы с крупномасштабными сетями.

К преимуществам данной программы (по отношению к известным) относятся:
* Использование эффективной структуры данных для хранения и работы с большими сетями;
* Построение LaTeX таблиц для представления результатов работы функций;
* Низкий порог входа для начала использования;
* Эффективное использование памяти с помощью принудительного удаления неиспользуемых объектов;
* Архитектура программы позволяет легко добавлять новые функции или модули для расширения её функциональности.

### ENG:
Abstract: The «NetLinkAnalyze» program is designed to calculate the characteristic description of connections in large-scale networks for further training the classifier and predicting the appearance of new connections at subsequent points in time, as well as to calculate the characteristics of static graphs, including the number of vertices and edges, density, components of weak connectivity, radius, diameter of the network and the coefficient of assortativity. It provides results in a format convenient for analysis - LaTeX tables with efficient use of the memory of a computing device for working with large-scale networks.

The advantages of this program (in relation to the known ones) include:
* Using an efficient data structure for storing and working with large networks;
* Building LaTeX tables to represent the results of functions;
* Low entry threshold to start using;
* Efficient use of memory by forcibly deleting unused objects;
* The architecture of the program makes it easy to add new functions or modules to expand its functionality.

## Documentation/Документация:
https://romiconez.github.io/CFLG/

## How to use:

(Data: https://github.com/RomiconEZ/CFLG/tree/main/tests/email-Eu-core-temporal-Dept3)

    pip install NetLinkAnalyze
    --------------------------
    from pathlib import Path
    import pandas as pd
    from IPython.core.display_functions import display
    from sklearn import linear_model, pipeline, preprocessing
    from cflg import graph_features_auc_score_tables, features_for_edges_of_static_graph

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
        
        (
            latex_feature_network_table_1,
            latex_feature_network_table_2,
            latex_feature_network_table_3,
            latex_feature_network_table_4,
            latex_auc_table,
        ) = graph_features_auc_score_tables(datasets_info, cls_model=cls_model, verbose=True)
       
        print(latex_feature_network_table_1)
        print(latex_feature_network_table_2)
        print(latex_feature_network_table_3)
        print(latex_feature_network_table_4)
        print(latex_auc_table)
        return


    def test_features_for_static_graph() -> None:
        def display_dataframe(df):
            with pd.option_context("display.max_columns", None):  # Показать все колонки
                display(df.head(5))  # Вывести первые 5 строк
    
        current_path = Path(__file__).parent
        name = "email-Eu-core-temporal-Dept3"
        path_to_data = str(current_path / name / f"out.{name}")
    
        X = features_for_edges_of_static_graph(path_to_data, verbose=True)
    
        return