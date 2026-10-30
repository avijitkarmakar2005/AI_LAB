import networkx as nx
import matplotlib.pyplot as plt

semantic_network = {
    "Rahul": [
        ("is-a", "student"),
        ("studies", "Python")
    ],
    "Python": [
        ("is-a", "Programming language"),
        ("used_for", "AI")
    ],
    "AI": [
        ("is-a", "Technology")
    ],
    "Dr.Sen": [
        ("teaches", "AI")
    ]
}


def display_network():
    print("\n--- Semantic Network ---")
    for concept, relations in semantic_network.items():
        for relation, target in relations:
            print(f"{concept} -- {relation} --> {target}")


def find_relationship(concept):
    if concept in semantic_network:
        print(f"\nRelationship of {concept}:")
        for relation, target in semantic_network[concept]:
            print(f"{concept} -- {relation} --> {target}")
    else:
        print("Concept not found")


def draw_graph():
    G = nx.DiGraph()

    for concept, relations in semantic_network.items():
        for relation, target in relations:
            G.add_edge(concept, target, label=relation)

    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        arrows=True,
        node_color="lightblue"
    )

    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=9
    )

    plt.title("Semantic Network")
    plt.show()


display_network()

while True:
    print("\n1. Search Concept")
    print("2. Exit")
    print("3. Draw Graph")

    choice = input("Enter your choice: ")

    if choice == "1":
        concept = input("Enter concept: ")
        find_relationship(concept)

    elif choice == "2":
        print("Program ended.")
        break

    elif choice == "3":
        draw_graph()

    else:
        print("Invalid choice.")