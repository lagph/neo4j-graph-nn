from graph_dbs import graph
import graph2np as gn
import pandas as pd
import numpy as np


def test_get_book_data_columns():
    authors = gn.get_authors(graph)
    results, ids = gn.get_book_data_columns(graph, authors)
    for i in range(len(authors)):
        author_id = authors.loc[i, ['id']][0]
        result = graph.run('MATCH (b:Book)-[WROTE]-(:Author {id: %s}) RETURN'
                           ' b.topic as topic, b.sales as sales, b.id as id' % author_id)
        assert all(result.to_data_frame() == results[i])


def test_book_data_columns_to_array():
    data_columns = [pd.DataFrame({'topic': ['topic_A', 'topic_B', 'topic_A'],
                                  'sales': [0, 0.23, 2.3],
                                  'idt': [0, 1, 2]}),
                    pd.DataFrame({'topic': ['topic_C', 'topic_D', 'topic_D'],
                                  'sales': [0, 1.23, 2.2],
                                  'idt': [2, 3, 26]})]
    author_ids = [5, 2]
    ids, X, y = gn.book_data_columns_to_array(data_columns, author_ids)
    author_a_feats = [2, 1, 0, 0, 2.3, 0.23, 0, 0]
    author_b_feats = [0, 0, 1, 2, 0, 0, 0, 3.43]
    assert ids == ((5, 0), (5, 1), (5, 2), (2, 2), (2, 3), (2, 26))
    assert (X == np.array([[1, 0, 0, 0] + author_a_feats, [0, 1, 0, 0] + author_a_feats,
                           [1, 0, 0, 0] + author_a_feats, [0, 0, 1, 0] + author_b_feats,
                           [0, 0, 0, 1] + author_b_feats, [0, 0, 0, 1] + author_b_feats])).all()
    assert (y == np.array([0, 0.23, 2.3, 0, 1.23, 2.2])).all()
