"""
In this script, we train a distributed XGBoost multi-class classification model using ECFP4 fingerprints.
Model hyperparameters are optimized via Bayesian Optimization.
This script uses a single node with multiple GPUs through a LocalCUDACluster.
"""

import pandas as pd
import numpy as np
import dask.array as da
import json
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from xgboost import dask as dxgb
from bayes_opt import BayesianOptimization

# ---- Configuration ----

stratification_type = 'bio_or_chem'  # choose from 'bio_or_chem' or 'specific_rule'

training_features_path = f'../data/training/reactant_ecfp4_fingerprints_stratified_by_{stratification_type}.parquet'
training_labels_path = f'../data/training/template_labels_stratified_by_{stratification_type}.parquet'

validation_features_path = f'../data/validation/reactant_ecfp4_fingerprints_stratified_by_{stratification_type}.parquet'
validation_labels_path = f'../data/validation/template_labels_stratified_by_{stratification_type}.parquet'

SAVE_MODEL_PATH = "../models/template_prioritizer_XGBoost_model.json"
SAVE_BEST_PARAMS_PATH = "../models/best_xgboost_hyperparams.json"

NUM_CLASSES = 3927
RANDOM_STATE = 42

# ---- Helper functions ----

def load_features_labels_from_parquet(feature_path, label_path):
    """Load ECFP4 features and label indices from .parquet files."""
    X = pd.read_parquet(feature_path).values
    y = pd.read_parquet(label_path)["Label Index"].values
    return X, y

def start_dask_cluster():
    """Start a LocalCUDACluster for single-node multi-GPU training."""
    cluster = LocalCUDACluster()
    client = Client(cluster)
    return client

def create_dask_dmatrix(client, X, y):
    """Create a DaskDMatrix from features and labels with matching partitions."""
    X_dask = da.from_array(X, chunks="auto")
    y_dask = da.from_array(y, chunks=(X_dask.chunks[0],))
    return dxgb.DaskDMatrix(client, X_dask, y_dask)

def train_xgboost(client, dtrain, dval, params, num_boost_round=1000, early_stopping_rounds=30):
    """Train an XGBoost model with Dask."""
    evals = [(dtrain, 'train'), (dval, 'validation')]
    output = dxgb.train(
        client,
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    return output['booster'], output['history']

# ---- Main script ----
if __name__ == '__main__':

    # ---- Load data ----
    print("Loading training and validation data...")

    X_train, y_train = load_features_labels_from_parquet(training_features_path, training_labels_path)
    X_val, y_val = load_features_labels_from_parquet(validation_features_path, validation_labels_path)

    # ---- Start Dask cluster ----
    print("Starting Dask cluster...")
    client = start_dask_cluster()
    print(f"Dask dashboard available at: {client.dashboard_link}")

    # ---- Create Dask DMatrices ----
    print("Creating Dask DMatrices...")
    dtrain = create_dask_dmatrix(client, X_train, y_train)
    dval = create_dask_dmatrix(client, X_val, y_val)

    # ---- Define Bayesian Optimization Objective ----
    def xgb_val_accuracy(max_depth, learning_rate, subsample, colsample_bytree):
        """Objective function to maximize validation accuracy."""
        max_depth = int(max_depth)

        params = {
            'objective': 'multi:softprob',
            'num_class': NUM_CLASSES,
            'tree_method': 'gpu_hist',  # for GPU training
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eval_metric': 'merror',
            'verbosity': 0,
        }

        booster, history = train_xgboost(client, dtrain, dval, params)

        # Properly handle merror list-of-lists
        validation_merrors = []
        for k, v in history.items():
            if 'validation' in k:
                for sublist in v.values():
                    validation_merrors.extend(sublist)

        best_merror = min(validation_merrors)
        val_accuracy = 1.0 - best_merror
        return val_accuracy

    # ---- Set up and run Bayesian Optimization ----
    print("Starting Bayesian Optimization...")

    pbounds = {
        'max_depth': (4, 12),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    }

    optimizer = BayesianOptimization(
        f=xgb_val_accuracy,
        pbounds=pbounds,
        random_state=RANDOM_STATE,
        verbose=2
    )

    optimizer.maximize(
        init_points=5,
        n_iter=25
    )

    # ---- Train final model with the best hyperparameters ----
    print("Training final model...")

    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])  # cast back to int

    # Save best hyperparameters
    with open(SAVE_BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best hyperparameters saved to {SAVE_BEST_PARAMS_PATH}.")

    final_params = {
        'objective': 'multi:softprob',
        'num_class': NUM_CLASSES,
        'tree_method': 'gpu_hist',
        'eval_metric': 'merror',
        'verbosity': 1,
        'max_depth': best_params['max_depth'],
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
    }

    final_booster, _ = train_xgboost(client, dtrain, dval, final_params, num_boost_round=2000, early_stopping_rounds=50)

    # ---- Save the final model ----
    print(f"Saving final model to {SAVE_MODEL_PATH}...")
    final_booster.save_model(SAVE_MODEL_PATH)

    print("Training and hyperparameter optimization complete.")

    # ---- Close Dask client ----
    client.close()
    print("Dask client closed. Job complete.")
