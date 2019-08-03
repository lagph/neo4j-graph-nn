from neo4j import GraphDatabase
from py2neo.data import Node, Relationship
from py2neo import Graph

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "test"))

graph = Graph(uri, auth=("neo4j", "test"))

a = Node("Person", name="Alice")
b = Node("Person", name="Bob")
c = Relationship(a, "KNOWS", b)

graph.create(a)
graph.create(b)
graph.create(c)
