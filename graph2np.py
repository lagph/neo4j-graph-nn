from typing import List, Tuple
import pandas as pd
from graph_dbs import Neo4jGraph
import numpy as np

IdTpl = Tuple[Tuple[int, int], ...]  # Data structure used to keep track of author and book ids


def get_authors(graph: Neo4jGraph) -> pd.DataFrame:
    query = "MATCH (n:Author) RETURN "
    query_suff = ','.join(["n.{} as {}".format(column_name, column_name)
                           for column_name in ("id", "country", "name")])
    query += query_suff
    return graph.run(query).to_data_frame()


def get_book_data_columns(graph: Neo4jGraph,
                          authors: pd.DataFrame) -> Tuple[List[pd.DataFrame], Tuple[int, ...]]:
    results, ids = [], []
    for _, idt, country, name in authors.itertuples():
        querydict = 'name: "{}", id: {}, country: "{}"'.format(
            name, idt, country)
        querydict = '{' + querydict + '}'
        ret = 'b.topic as topic, b.sales as sales, b.id as id'
        result = graph.run('MATCH (b:Book)-[WROTE]-(a:Author {}) RETURN {}'.format(
                           querydict, ret))
        results.append(result.to_data_frame())
        ids.append(idt)
    return results, tuple(ids)


def book_data_columns_to_array(
    data_columns: List[pd.DataFrame], ids: Tuple) \
        -> Tuple[IdTpl, np.ndarray, np.ndarray]:
    topics = [np.array(series.topic) for series in data_columns]
    all_topics = np.concatenate(topics)
    topics = sorted(np.unique(all_topics))
    X = np.zeros((len(all_topics), 3 * len(topics)))
    y = []
    encoding = {topic: i for i, topic in enumerate(topics)}
    idx = 0
    idlist = []
    for author_idx, author_df in enumerate(data_columns):
        topics = np.unique(author_df.topic)
        topic_features = {}
        for topic in topics:
            sales = author_df[author_df.topic == topic]['sales']
            topic_features[topic] = {'num_books': len(
                sales), 'total_sales': sum(sales)}
        for _, topic, sales, idt in author_df.itertuples():
            X[idx, encoding[topic]] = 1
            for tpc in topics:
                X[idx, len(encoding) + encoding[tpc]
                  ] = topic_features[tpc]['num_books']
                X[idx, 2 * len(encoding) + encoding[tpc]
                  ] = topic_features[tpc]['total_sales']
            idx += 1
            idlist.append((ids[author_idx], idt))
            y.append(sales)
    return tuple(idlist), X, np.array(y)


def graph_to_numpy(graph: Neo4jGraph) -> Tuple[IdTpl, np.ndarray, np.ndarray]:
    authors = get_authors(graph)
    data_cols, ids = get_book_data_columns(graph, authors)
    return book_data_columns_to_array(data_cols, ids)
