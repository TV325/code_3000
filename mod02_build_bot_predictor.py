# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.0000001,
        n_estimators=500,
        max_depth=None,
        subsample=0.00001,
        min_samples_leaf=3,
        random_state=seed
    )
    model.fit(X, y)
    return model