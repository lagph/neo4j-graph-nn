from py2neo import Graph, Node, Relationship
from typing import Tuple, Dict


class Neo4jGraph:
    def __init__(self,
                 uri: str,
                 auth: Tuple[str, str]
                 ):
        self._graph = Graph(uri=uri, auth=auth)

    def commit_relation(self,
                        src: Dict[str, str],
                        rel: Dict[str, str],
                        dst: Dict[str, str],
                        ) -> None:
        srckind = src['kind']
        srcnode = Node(
            srckind, **{k: v for k, v in src.items() if k != 'kind'})
        dstkind = dst['kind']
        dstnode = Node(
            dstkind, **{k: v for k, v in dst.items() if k != 'kind'})
        relkind = rel['kind']
        relationship = Relationship(srcnode, relkind, dstnode,
                                    **{k: v for k, v in rel.items() if k != 'kind'})
        self._graph.merge(srcnode, "Author", "name")
        self._graph.create(dstnode)
        self._graph.create(relationship)

    def run(self, query: str):
        return self._graph.run(query)


graph = Neo4jGraph("bolt://localhost:7687", auth=("neo4j", "test"))
