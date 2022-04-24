import gc

import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

from functions.feature_selection_and_normalization import UpperQuartile


def nondask_load_data(filepaths):
    return pd.read_csv(filepaths[0]), pd.read_csv(filepaths[1])


def feature_preprocessing(feature_matrix):
    uq = UpperQuartile()
    uq_feature_matrix = uq.fit_transform(feature_matrix)

    # For gene features to be maximally informative, we want them to have a minimum expression signal and to
    # vary at least a little across samples.
    mean = feature_matrix.mean(axis=0)
    var = feature_matrix.var(axis=0)

    threshold_feature_matrix = uq_feature_matrix[uq_feature_matrix.columns[(mean > mean.quantile(0.25)) &
                                                                           (var > var.quantile(0.25))]]

    log_scaled_feature_matrix = threshold_feature_matrix.applymap(lambda gene: np.log2(gene + 1))

    return log_scaled_feature_matrix


def pre_ml_processing(nondask_feature_array: np.array, nondask_label_array: np.array,
                      task: str='classification') -> tuple[np.array, np.array, np.array, np.array]:

    X_train, X_test, y_train, y_test = train_test_split(nondask_feature_array, nondask_label_array, test_size=0.3,
                                                        shuffle=True, random_state=42)

    if task.lower() == 'classification':
        enc = LabelEncoder()
        enc.fit(y_train)
        y_train_processed, y_test_processed = enc.transform(np.ravel(y_train)), enc.transform(np.ravel(y_test))
    elif task.lower() == 'regression':
        y_train_processed, y_test_processed = np.ravel(y_train), np.ravel(y_test)
    else:
        raise ValueError('Invalid ML task!')

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_scaled, X_test_scaled = sc.transform(X_train), sc.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_processed, y_test_processed

def nondask_default_xgboost_pipeline(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array,
                                     task: str='classification') -> tuple[float, float, float]:

    if task.lower() == 'classification':
        bst_cv = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        bst_eval = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
        scoring = 'accuracy'
    elif task.lower() == 'regression':
        bst_cv = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        bst_eval = xgb.XGBRegressor(random_state=42, eval_metric='rmse', n_jobs=-1)
        scoring = 'r2'
    else:
        raise ValueError('Invalid ML task!')

    cv_scores = cross_val_score(bst_cv, X_train, y_train, cv=10, scoring=scoring, n_jobs=-1)
    mean_cv_score, std_cv_score = np.mean(cv_scores), np.var(cv_scores)

    bst_eval.fit(X_train, y_train)
    prediction = bst_eval.predict(X_test)
    if task.lower() == 'classification':
        eval_score = accuracy_score(y_test, prediction)
    else:
        eval_score = r2_score(y_test, prediction)

    return mean_cv_score, std_cv_score, eval_score


def nondask_hpo_sgd_pipeline(param_dist, X_train, X_test, y_train, y_test):
    cv_bst = SGDClassifier(random_state=42, n_jobs=-1)
    cv_search_bst = RandomizedSearchCV(cv_bst, param_dist, n_iter=100, random_state=42, scoring='accuracy', cv=2,
                                       n_jobs=-1)
    cv_scores = cross_val_score(cv_search_bst, X_train, y_train, cv=5)
    mean_cv_score, std_cv_score = np.mean(cv_scores), np.var(cv_scores)

    bst = SGDClassifier(random_state=42, n_jobs=-1)
    search_bst = RandomizedSearchCV(bst, param_dist, n_iter=1000, random_state=42, scoring='accuracy',
                                    n_jobs=-1)
    search_bst.fit(X_train, y_train)
    prediction = search_bst.best_estimator_.predict(X_test)
    eval_score = accuracy_score(y_test, prediction)

    return mean_cv_score, std_cv_score, eval_score


def nondask_pipeline(filepaths: tuple=None, task: str='classification', param_dist: dict=None):
    nondask_feature_matrix, nondask_label_vector = nondask_load_data(filepaths)

    preprocessed_feature_matrix = feature_preprocessing(nondask_feature_matrix)
    gc.collect()

    X_train, X_test, y_train, y_test = pre_ml_processing(preprocessed_feature_matrix, nondask_label_vector, task)
    del preprocessed_feature_matrix, nondask_label_vector
    gc.collect()

    if param_dist:
        mean_cv_score, std_cv_score, eval_score = nondask_hpo_sgd_pipeline(param_dist, X_train, X_test, y_train,
                                                                               y_test)
    else:
        mean_cv_score, std_cv_score, eval_score = nondask_default_xgboost_pipeline(X_train, X_test, y_train, y_test,
                                                                                   task)

    return mean_cv_score, std_cv_score, eval_score