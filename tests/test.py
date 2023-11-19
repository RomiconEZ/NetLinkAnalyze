import sys

from cflg.src.Base import Edge, Node, SelectApproach, StaticGraph

test_graph_path = sys.argv[1]
output_path = sys.argv[2]

staticGraph = StaticGraph()

with open(test_graph_path) as edges:
    list_of_items = edges.read().split("\n")
    list_of_items.pop(-1)
    edge_number = 0
    for item in list_of_items:
        item = item.split(" ")
        if int(item[0]) == int(item[1]):
            continue
        staticGraph.add_edge(
            Edge(
                number=edge_number,
                start_node=Node(number=int(item[0]) - 1),
                end_node=Node(number=int(item[1]) - 1),
                timestamp=int(item[-1]),
            )
        )
        edge_number += 1

snowball_sample_approach = SelectApproach(1, 2)
random_selected_vertices_approach = SelectApproach()

with open(output_path, "w") as out:
    out.writelines(f"Количество вершин в графе: {staticGraph.count_vertices()}\n")
    out.writelines(f"Количество рёбер в графе: {staticGraph.count_edges()}\n")
    out.writelines(f"Плотность графа: {staticGraph.density()}\n")
    out.writelines(f"Доля вершин: {staticGraph.share_of_vertices()}\n")
    out.writelines(f"Количество компонент слабой связности: {staticGraph.get_number_of_connected_components()}\n")

    out.writelines(
        f"Количество вершин в наибольшей компоненте слабой связности: {staticGraph.get_largest_connected_component().count_vertices()}\n"
    )
    out.writelines(
        f"Количество рёбер в наибольшей компоненте слабой связности: {staticGraph.get_largest_connected_component().count_edges()}\n"
    )

    sg = snowball_sample_approach(staticGraph.get_largest_connected_component())
    out.writelines(f"Оценка радиуса графа (СК): {staticGraph.get_radius(sg)}\n")
    out.writelines(f"Оценка диаметра графа (СК): {staticGraph.get_diameter(sg)}\n")
    out.writelines(
        f"Оценка 90 процентиля расстояния между вершинами графа (СК): {staticGraph.percentile_distance(sg)}\n"
    )

    sg = random_selected_vertices_approach(staticGraph.get_largest_connected_component())
    out.writelines(f"Оценка радиуса графа (СВВ): {staticGraph.get_radius(sg)}\n")
    out.writelines(f"Оценка диаметра графа (СВВ): {staticGraph.get_diameter(sg)}\n")
    out.writelines(
        f"Оценка 90 процентиля расстояния между вершинами графа (СВВ): {staticGraph.percentile_distance(sg)}\n"
    )

    out.writelines(f"Коэффициент ассортативности: {staticGraph.assortative_factor()}\n")

    out.writelines(f"Средний кластерный коэффициент сети: {staticGraph.average_cluster_factor()}\n")
