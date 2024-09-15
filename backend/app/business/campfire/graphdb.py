import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def to_cypher_string(props: dict, suffix="") -> str:
    if not props:
        return ""
    return "{" + ", ".join([f"{k}: ${k}{suffix}" for k in props.keys()]) + "}"


def to_cypher_params(props: dict, suffix="") -> dict:
    if not props:
        return {}
    return {k + suffix: v for k, v in props.items()}


class GraphDb(object):
    def __init__(self):
        self.user = os.environ["neo4j_user"]
        self.pswd = os.environ["neo4j_pswd"]
        self.driver = GraphDatabase.driver(
            uri="bolt://neo4j", auth=(self.user, self.pswd)
        )

    def add_node(self, label: str, **props):
        def _add_node(tx, label, **props):
            properties = to_cypher_string(props)
            query = f"CREATE (n: {label} {properties})"
            tx.run(query, **props)

        with self.driver.session() as session:
            session.execute_write(_add_node, label, **props)

    def merge_node(self, label: str, **props):
        def _merge_node(tx, label, **props):
            properties = to_cypher_string(props)
            query = f"MERGE (n: {label} {properties})"
            tx.run(query, **props)

        with self.driver.session() as session:
            session.execute_write(_merge_node, label, **props)

    def create_index(self, label: str, keys: list[str]):
        def _create_index(tx, label: str, keys: list[str]):
            # make csv text with `n.key_name`
            keys_text = ", ".join([f"n.{key}" for key in keys])
            index_name = f"index_node_{label.lower()}_{keys_text.replace(', ', '_').replace('n.', '')}"
            query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n: {label}) ON ({keys_text})"
            tx.run(query)

        with self.driver.session() as session:
            session.execute_write(_create_index, label, keys)

    def add_edge(
        self, label: str = "Edge", node_keys_src: dict = {}, node_keys_trg: dict = {}
    ):
        """
        Adds an edge between two nodes in the graph.

        Args:
            label (str): The label of the edge. Defaults to "Edge".
            node_keys_src (dict): The properties of the source node. If no label is provided, it defaults to "Node".
            node_keys_trg (dict): The properties of the target node. If no label is provided, it defaults to "Node".
        """

        def _add_edge(tx, label: str, node_keys_src: dict, node_keys_trg: dict):
            node_label_src = node_keys_src.pop("label", "Node")
            node_label_trg = node_keys_trg.pop("label", "Node")
            node_cond_src = to_cypher_string(node_keys_src, suffix="_src")
            node_cond_trg = to_cypher_string(node_keys_trg, suffix="_trg")
            query = f"""
            MATCH (node_src:{node_label_src} {node_cond_src}), (node_trg:{node_label_trg} {node_cond_trg})
            CREATE (node_src)-[r:{label}]->(node_trg)
            """
            _node_keys_src = to_cypher_params(node_keys_src, suffix="_src")
            _node_keys_trg = to_cypher_params(node_keys_trg, suffix="_trg")
            tx.run(query, **_node_keys_src, **_node_keys_trg)

        with self.driver.session() as session:
            session.execute_write(_add_edge, label, node_keys_src, node_keys_trg)

    def merge_edge(
        self, label: str = "Edge", node_keys_src: dict = {}, node_keys_trg: dict = {}
    ):
        """
        Merges an edge between two nodes in the graph.

        Args:
            label (str): The label of the edge. Defaults to "Edge".
            node_keys_src (dict): The properties of the source node. If no label is provided, it defaults to "Node".
            node_keys_trg (dict): The properties of the target node. If no label is provided, it defaults to "Node".
        """

        def _merge_edge(tx, label: str, node_keys_src: dict, node_keys_trg: dict):
            node_label_src = node_keys_src.pop("label", "Node")
            node_label_trg = node_keys_trg.pop("label", "Node")
            node_cond_src = to_cypher_string(node_keys_src, suffix="_src")
            node_cond_trg = to_cypher_string(node_keys_trg, suffix="_trg")
            query = f"""
            MATCH (node_src:{node_label_src} {node_cond_src}), (node_trg:{node_label_trg} {node_cond_trg})
            MERGE (node_src)-[r:{label}]->(node_trg)
            """
            _node_keys_src = to_cypher_params(node_keys_src, suffix="_src")
            _node_keys_trg = to_cypher_params(node_keys_trg, suffix="_trg")
            tx.run(query, **_node_keys_src, **_node_keys_trg)

        with self.driver.session() as session:
            session.execute_write(_merge_edge, label, node_keys_src, node_keys_trg)

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    g = GraphDb()
    g.add_node(
        label="TestLabel",
        url="http://localhost:8397/hello",
        name="test1",
        area="area1",
        value="value1",
    )
    g.add_node(
        label="TestLabel",
        url="http://localhost2:8397/world",
        name="test2",
        area="area2",
        value="value2",
    )

    g.add_edge(
        label="RELATION",
        node_keys_src={"label": "TestLabel", "name": "test1"},
        node_keys_trg={"label": "TestLabel", "name": "test2"},
    )
    g.close()
    print("done!")
