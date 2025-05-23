"""
In this script, we train a distributed XGBoost multi-class classification model using ECFP4 fingerprints.
Ray is used
"""

import pickle
import json
import numpy as np
import ray
from xgboost_ray import RayDMatrix, RayParams, train, RayXGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
import pandas as pd

# check to see if Ray can see GPUs
import subprocess
print("[DEBUG] nvidia-smi output:")
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

# specify Ray configurations for distributed multi-node, multi-gpu training
# for bash submission script, number GPUs requested = num_actors x gpus_per_actor
# number of CPUs requested = num_actors x cpus_per_actor
max_actor_restarts: int = 2 # number of times Ray will restart an actor if it fails
num_actors: int = 8 # total number of parallel workers (ideally = number of GPUs)
cpus_per_actor: int = 4 # number of CPUs assigned per actor
gpus_per_actor: int = 1 # number of GPUs per actor (must be >=1 with 'tree_method' set to 'gpu_hist')

# load in fingerprints and labels from training and validation datasets
training_fps_path = f'../data/training/training_reactant_ecfp4_fingerprints.parquet'
training_labels_path = f'../data/training/training_template_labels.parquet'

validation_fps_path = f'../data/validation/validation_reactant_ecfp4_fingerprints.parquet'
validation_labels_path = f'../data/validation/validation_template_labels.parquet'

# define output filepath for multi-class template prioritizer model
model_output_filepath = f"../models/template_prioritizer_baseline_model_XGBoost.pkl"
opt_params_filepath = f'../models/template_prioritizer_baseline_XGBoost.json'

X_train = pd.read_parquet(training_fps_path).to_numpy()
y_train = pd.read_parquet(training_labels_path).to_numpy().flatten()

X_val = pd.read_parquet(validation_fps_path).to_numpy()
y_val = pd.read_parquet(validation_labels_path).to_numpy().flatten()

def XGBC_objective(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   max_actor_restarts: int,
                   num_actors: int,
                   cpus_per_actor: int,
                   gpus_per_actor: int,):
    """
    Objective function for XGBoost hyperparameter optimization via a Bayesian optimization procedure.
    This function will be passed to an instantiated BayesianOptimization object
    """

    def objective(learning_rate: float,
                  max_leaves: int,
                  max_depth: int,
                  reg_alpha: float,
                  reg_lambda: float,
                  n_estimators: int,
                  min_child_weight: float,
                  colsample_bytree: float,
                  colsample_bylevel: float,
                  colsample_bynode: float,
                  subsample: float,
                  scale_pos_weight: float) -> float:

        params = {'learning_rate': learning_rate,
                  'max_leaves': int(max_leaves),
                  'max_depth': int(max_depth),
                  'reg_alpha': reg_alpha,
                  'reg_lambda': reg_lambda,
                  'n_estimators': int(n_estimators),
                  'min_child_weight': min_child_weight,
                  'colsample_bytree': colsample_bytree,
                  'colsample_bylevel': colsample_bylevel,
                  'colsample_bynode': colsample_bynode,
                  'subsample': subsample,
                  'scale_pos_weight': scale_pos_weight,
                  'objective': 'multi:softprob',
                  'eval_metric': 'logloss',
                  'tree_method': 'gpu_hist', # since we have XGBoost 1.6.2
                  'random_state': 42}

        # train XGBoost classifier on training data
        model = RayXGBClassifier(**params)

        ray_params = RayParams(max_actor_restarts = max_actor_restarts,
                               num_actors = num_actors,
                               cpus_per_actor = cpus_per_actor,
                               gpus_per_actor = gpus_per_actor)

        model.fit(X_train, y_train, ray_params = ray_params)

        # then evaluate on validation data by predicting the template label using validation fingerprints
        y_val_predicted = model.predict(X_val)

        # finally, calculate the AUPRC score between the validation labels and the validation predicted probabilities
        accuracy = accuracy_score(y_true = y_val,
                                  y_pred = y_val_predicted)
        return accuracy

    return objective

def run_bayesian_hyperparameter_search(X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_val: np.ndarray,
                                       max_actor_restarts: int,
                                       num_actors: int,
                                       cpus_per_actor: int,
                                       gpus_per_actor: int,):

    objective = XGBC_objective(X_train, y_train, X_val, y_val,
                               max_actor_restarts, num_actors, cpus_per_actor, gpus_per_actor)

    # Define the bounds for each hyperparameter
    pbounds = {
        'learning_rate': (0.1, 0.5),
        'max_leaves': (20, 300),
        'max_depth': (1, 15),
        'reg_alpha': (0, 1.0),
        'reg_lambda': (0, 1.0),
        'n_estimators': (20, 300),
        'min_child_weight': (2, 10),
        'colsample_bytree': (0.5, 1.0),
        'colsample_bylevel': (0.5, 1.0),
        'colsample_bynode': (0.5, 1.0),
        'subsample': (0.4, 1.0),
        'scale_pos_weight': (1, 5)
    }

    optimizer = BayesianOptimization(f = objective,
                                     pbounds = pbounds,
                                     random_state = 42)

    optimizer.maximize(
        init_points = 5,  # number of randomly chosen points to sample the target function before fitting the GP
        n_iter = 20)  # total number of times the process is to be repeated

    best_params = optimizer.max['params']
    best_score = optimizer.max['target']

    print(f"Best accuracy: {best_score:.4f} achieved with {best_params}")

    return best_params

opt_hyperparams = run_bayesian_hyperparameter_search(X_train, y_train, X_val, y_val,
                                                     max_actor_restarts, num_actors, cpus_per_actor, gpus_per_actor)

# save the optimized hyperparameters to a json file
with open(opt_params_filepath,'w') as json_file:
    json.dump(opt_hyperparams, json_file)

# with the Bayesian optimized hyperparameters, train the baseline model

molecular_classifier_xgboost = RayXGBClassifier(objective = 'multi:softprob',
                                                random_state = 42,
                                                max_leaves = int(opt_hyperparams['max_leaves']),
                                                learning_rate = opt_hyperparams['learning_rate'],
                                                max_depth = int(opt_hyperparams['max_depth']),
                                                reg_alpha = opt_hyperparams['reg_alpha'],
                                                reg_lambda = opt_hyperparams['reg_lambda'],
                                                n_estimators = int(opt_hyperparams['n_estimators']),
                                                min_child_weight = opt_hyperparams['min_child_weight'],
                                                colsample_bytree = opt_hyperparams['colsample_bytree'],
                                                colsample_bylevel = opt_hyperparams['colsample_bylevel'],
                                                colsample_bynode = opt_hyperparams['colsample_bynode'],
                                                subsample = opt_hyperparams['subsample'],
                                                scale_pos_weight = opt_hyperparams['scale_pos_weight'],
                                                tree_method = 'gpu_hist')

ray_params = RayParams(max_actor_restarts =max_actor_restarts,
                       num_actors = num_actors,
                       cpus_per_actor = cpus_per_actor,
                       gpus_per_actor = gpus_per_actor)

molecular_classifier_xgboost.fit(X_train, y_train, ray_params = ray_params)

# save the trained XGBoost model with pickle
with open(model_output_filepath, 'wb') as model_file:
    pickle.dump(molecular_classifier_xgboost, model_file)