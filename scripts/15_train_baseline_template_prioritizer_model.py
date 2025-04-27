"""
In this script, we train a distributed XGBoost multi-class classification model with ecfp4 fingerprints.
Model hyperparameters here are optimized using a Bayesian hyperparameter optimization procedure.
The models here will form a baseline with which we can continue trying different architectures for improvement.
"""
from dask.distributed import Client
# from dask_cuda import LocalCUDACluster # use this for single-node, multi-GPU
from dask_mpi import initialize # use this for multi-node, multi-GPU
from xgboost import dask as dxgb
from bayes_opt import BayesianOptimization
import dask.array as da
import json
import pandas as pd
import numpy as np

# ---- Configuration ----

stratification_type = 'bio_or_chem' # choose from 'bio_or_chem' or 'specific_rule'

training_features_path = f'../data/training/reactant_ecfp4_fingerprints_stratified_by_{stratification_type}.parquet'
training_labels_path = f'../data/training/template_labels_stratified_by_{stratification_type}.parquet'

validation_features_path = f'../data/validation/reactant_ecfp4_fingerprints_stratified_by_{stratification_type}.parquet'
validation_labels_path = f'../data/validation/template_labels_stratified_by_{stratification_type}.parquet'

SAVE_MODEL_PATH = "../models/template_prioritizer_XGBoost_model.json"

NUM_CLASSES = 3927
N_GPUS = 4  # adjust this depending on how many GPUs you want to use
RANDOM_STATE = 42

# ---- Helper functions ----

def load_features_labels_from_parquet(feature_path, label_path):
    X = pd.read_parquet(feature_path).values
    y = pd.read_parquet(label_path)["Label Index"].values
    return X, y

###### use the following helper-function for single-node, multi-gpu
# def start_dask_cluster(n_gpus):
#     cluster = LocalCUDACluster(n_workers=n_gpus, threads_per_worker=4)
#     client = Client(cluster)
#     return client

###### use the following helper-function for multi-node, multi-gpu
def start_dask_cluster():
    initialize()
    client = Client()
    return client

def create_dask_dmatrix(client, X, y, chunks_per_gpu):
    X_dask = da.from_array(X, chunks=(chunks_per_gpu, -1))
    y_dask = da.from_array(y, chunks=(chunks_per_gpu,))
    return dxgb.DaskDMatrix(client, X_dask, y_dask)

def train_xgboost(client, dtrain, dval, params, num_boost_round=1000, early_stopping_rounds=30):

    evals = [(dtrain, 'train'), (dval, 'validation')]
    output = dxgb.train(
                        client,
                        params,
                        dtrain,
                        num_boost_round = num_boost_round,
                        evals = evals,
                        early_stopping_rounds = early_stopping_rounds,
                        verbose_eval = False)

    return output['booster'], output['history']

# ---- Load data ----
print("Loading data...")

X_train, y_train = load_features_labels_from_parquet(
    feature_path = training_features_path,
    label_path = training_labels_path)

X_val, y_val = load_features_labels_from_parquet(
    feature_path = validation_features_path,
    label_path = validation_labels_path)

# ---- Start Dask cluster ----
print("Starting Dask cluster...")
#client = start_dask_cluster(N_GPUS) # use this for single node, multi-GPU
client = start_dask_cluster()
print(f"Dask dashboard running at: {client.dashboard_link}")

# Create Dask DMatrices
chunks_per_gpu = len(X_train) // N_GPUS
dtrain = create_dask_dmatrix(client, X_train, y_train, chunks_per_gpu)
dval = create_dask_dmatrix(client, X_val, y_val, chunks_per_gpu)


# ---- Define Bayesian Optimization Objective ----
def xgb_val_accuracy(max_depth, learning_rate, subsample, colsample_bytree):

    max_depth = int(max_depth)

    params = {
        'objective': 'multi:softprob',
        'num_class': NUM_CLASSES,
        'tree_method': 'gpu_hist', # use gpu_hist for XGBoost 1.6.2
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'eval_metric': 'merror',
        'verbosity': 0,
    }

    booster, history = train_xgboost(client, dtrain, dval, params)

    # get the best validation accuracy (1 - merror)
    best_merror = np.min([history[key] for key in history if 'validation' in key][0])
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

with open("../models/best_xgboost_hyperparams.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best hyperparameters saved!")

final_params = {
    'objective': 'multi:softprob',
    'num_class': NUM_CLASSES,
    'tree_method': 'gpu_hist', # use gpu_hist for XGBoost 1.6.2
    'eval_metric': 'merror',
    'verbosity': 1,
    'max_depth': best_params['max_depth'],
    'learning_rate': best_params['learning_rate'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
}

final_booster, _ = train_xgboost(client, dtrain, dval, final_params, num_boost_round = 2000, early_stopping_rounds=50)

# ---- Save the final model ----
print(f"Saving best model to {SAVE_MODEL_PATH}...")
final_booster.save_model(SAVE_MODEL_PATH)

print("Training and hyperparameter optimization complete!")