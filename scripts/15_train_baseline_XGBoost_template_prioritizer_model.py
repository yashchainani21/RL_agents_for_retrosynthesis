"""
In this script, we train a distributed XGBoost multi-class classification model using ECFP4 fingerprints.
Model hyperparameters are optimized via Bayesian Optimization.
This script uses a single node with multiple GPUs through a LocalCUDACluster.
"""

import pickle
import json
import numpy as np
import ray
from xgboost_ray import RayDMatrix, RayParams, train, RayXGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import average_precision_score
import pandas as pd

# load in fingerprints and labels from training and validation datasets
training_features_path = f'../data/training/training_reactant_ecfp4_fingerprints.parquet'
training_labels_path = f'../data/training/training_template_labels.parquet'

validation_features_path = f'../data/validation/validation_reactant_ecfp4_fingerprints.parquet'
validation_labels_path = f'../data/validation/validation_template_labels.parquet'

SAVE_MODEL_PATH = "./models/template_prioritizer_XGBoost_model.json"
SAVE_BEST_PARAMS_PATH = "./models/best_xgboost_hyperparams.json"

# ---- Helper functions ----



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
        'colsample_bytree': (0.5, 1.0)}

    optimizer = BayesianOptimization(
        f=xgb_val_accuracy,
        pbounds=pbounds,
        random_state=RANDOM_STATE,
        verbose=2)

    optimizer.maximize(
        init_points=5,
        n_iter=25)

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
