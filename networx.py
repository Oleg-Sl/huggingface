import networkx as nx
from itertools import combinations

# Создаем граф
G = nx.Graph()

# Добавляем ребра (пары)
edges = [
    ("отправитель", "человек1"),
    ("человек1", "человек2"), 
    ("человек3", "человек4"),
    ("человек4", "получатель")
]
G.add_edges_from(edges)

# Функция для поиска путей с учетом возможных совпадений имен
def find_paths_with_collisions(G, sender, receiver, max_intermediate=2):
    # Получаем все уникальные узлы
    nodes = list(G.nodes())
    
    # Создаем словарь для отображения имен на множество индексов
    name_to_nodes = {}
    for node in nodes:
        name_to_nodes.setdefault(node, []).append(node)
    
    # Проверяем возможные совпадения (предполагаем, что совпадать могут только люди 1-4)
    people_nodes = [n for n in nodes if n.startswith("человек")]
    
    paths = []
    
    # Прямой путь (0 промежуточных)
    if G.has_edge(sender, receiver):
        paths.append([sender, receiver])
    
    # Через одного промежуточного
    for intermediate in nodes:
        if intermediate not in [sender, receiver]:
            if G.has_edge(sender, intermediate) and G.has_edge(intermediate, receiver):
                paths.append([sender, intermediate, receiver])
    
    # Через двух промежуточных
    for node1 in nodes:
        for node2 in nodes:
            if (node1 not in [sender, receiver] and node2 not in [sender, receiver] 
                    and node1 != node2):
                if (G.has_edge(sender, node1) and G.has_edge(node1, node2) 
                        and G.has_edge(node2, receiver)):
                    paths.append([sender, node1, node2, receiver])
    
    # Теперь проверяем возможные совпадения имен
    # Генерируем все возможные отображения совпадений
    people_combinations = list(combinations(people_nodes, 2))
    
    for combo in people_combinations:
        node1, node2 = combo
        
        # Создаем модифицированный граф с учетом совпадения
        H = G.copy()
        
        # Контрактируем узлы (объединяем их)
        # В NetworkX нет прямой функции контракции, поэтому эмулируем
        # Создаем новый узел с объединенным именем
        merged_name = f"{node1}/{node2}"
        
        # Добавляем все связи обоих узлов к новому узлу
        neighbors = set(H.neighbors(node1)) | set(H.neighbors(node2))
        
        # Удаляем старые узлы
        H.remove_node(node1)
        H.remove_node(node2)
        
        # Добавляем новый объединенный узел
        H.add_node(merged_name)
        for neighbor in neighbors:
            if neighbor != merged_name:
                H.add_edge(merged_name, neighbor)
        
        # Ищем пути в модифицированном графе
        try:
            # Все простые пути длиной до 4 узлов
            all_paths = nx.all_simple_paths(H, source=sender, target=receiver, cutoff=4)
            for path in all_paths:
                if 2 <= len(path) <= 4:  # 2-4 узла (0-2 промежуточных)
                    # Заменяем объединенное имя на оригинальные для ясности
                    decoded_path = []
                    for node in path:
                        if node == merged_name:
                            decoded_path.append(f"{node1}={node2}")
                        else:
                            decoded_path.append(node)
                    if decoded_path not in paths:
                        paths.append(decoded_path)
        except nx.NetworkXNoPath:
            pass
    
    return paths

# Использование
sender = "отправитель"
receiver = "получатель"
paths = find_paths_with_collisions(G, sender, receiver)

print("Найденные пути:")
for i, path in enumerate(paths, 1):
    print(f"{i}. {' → '.join(path)}")