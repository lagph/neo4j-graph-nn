from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from pprint import pprint
import json
from os.path import join
from query2np import graph_to_numpy
from graph_dbs import graph
from argparse import ArgumentParser


def compute_baseline_score(X: np.ndarray, y: np.ndarray):
    models = {
        "GradientBoosting":
            {
                "estimator": GradientBoostingRegressor,
                "param_distributions":
                {
                    "n_estimators": [50, 100, 200, 500],
                    "max_depth": [2, 3, 4, 5],
                    "subsample": [1.0, 0.6],
                }
            },
        "RandomForest":
            {
                "estimator": RandomForestRegressor,
                "param_distributions":
                {
                    "n_estimators": [50, 100, 200, 500],
                    "min_samples_split": [2, 3, 4],
                    "max_depth": [2, 3, 4, 5],
                    "min_samples_leaf": [1, 2],
                }
            },
        "RidgeRegression":
            {
                "estimator": Ridge,
                "param_distributions":
                {
                    "alpha": [1, 4, 0.5]
                }
            }
    }
    results = {}
    for name, model in models.items():
        print("evaluating {}...".format(name))
        estimator = model['estimator']()
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=model['param_distributions'],
            n_iter=20,
            cv=5,
            iid=False,
        )
        search.fit(X, y)
        score = search.score(X, y)
        results[name] = score
        print("done, {} scored {}".format(name, score))
    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log-dir')
    args = parser.parse_args()
    _, X, y = graph_to_numpy(graph)
    scores = compute_baseline_score(X, y)
    pprint(scores)
    if args.log_dir is not None:
        json.dump(scores, open(join(args.log_dir, 'baseline_scores.json', 'w')))
