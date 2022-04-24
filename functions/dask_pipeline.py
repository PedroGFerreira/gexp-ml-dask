import gc

from sklearn.linear_model import SGDClassifier
import dask.array as da
import dask.dataframe as dd
import numpy as np
import xgboost as xgb

from dask_ml.metrics import accuracy_score, r2_score
from dask_ml.model_selection import train_test_split, RandomizedSearchCV
from dask_ml.preprocessing import LabelEncoder, StandardScaler

from functions.feature_selection_and_normalization import UpperQuartileDask
from functions.dask_cv import dask_cv


def dask_load_data(filepaths: tuple[str, str]) -> tuple[dd.DataFrame, dd.Series]:
    if 'parquet' in filepaths[0]:
        dask_feature_matrix = dd.read_parquet(filepaths[0]).persist()
    else:
        dask_feature_matrix = dd.read_csv(filepaths[0], assume_missing=True, sample=2000000)
    if 'parquet' in filepaths[1]:
        dask_label_vector = dd.read_parquet(filepaths[1])
    else:
        dask_label_vector = dd.read_csv(filepaths[1])

    return dask_feature_matrix, dask_label_vector


def feature_preprocessing(feature_matrix: dd.DataFrame) -> dd.DataFrame:
    uq = UpperQuartileDask()
    uq_feature_matrix = uq.fit_transform(feature_matrix).persist()

    # For gene features to be maximally informative, we want them to have a minimum expression signal and to
    # vary at least a little across samples.
    mean = uq_feature_matrix.mean(axis=0).persist()
    var = uq_feature_matrix.var(axis=0).persist()

    threshold_feature_matrix = uq_feature_matrix[uq_feature_matrix.columns[(mean > mean.quantile(0.25)) &
                                                                           (var > var.quantile(0.25))]
    ].repartition(partition_size='64MB')

    log_scaled_feature_matrix = threshold_feature_matrix.applymap(lambda gene: da.log2(gene + 1))

    return log_scaled_feature_matrix


def df_to_array(preprocessed_feature_matrix: dd.DataFrame, dask_label_vector: dd.Series) -> tuple[da.Array, da.Array]:
    preprocessed_feature_array = preprocessed_feature_matrix.to_dask_array(lengths=True).rechunk('auto')
    label_array = dask_label_vector.to_dask_array(lengths=True).rechunk((preprocessed_feature_array.chunks[0], None))

    return preprocessed_feature_array, label_array


def pre_ml_processing(dask_feature_array: da.Array, dask_label_array: da.Array,
                      task: str='classification') -> tuple[da.Array, da.Array, da.Array, da.Array]:
    X_train, X_test, y_train, y_test = train_test_split(dask_feature_array, dask_label_array, test_size=0.3,
                                                        shuffle=True, random_state=42)
    y_train, y_test = da.ravel(y_train), da.ravel(y_test)

    # Persists are very important from here on out because we use these X and y training/test arrays a lot.

    if task.lower() == 'classification':
        enc = LabelEncoder()
        enc.fit(y_train)
        y_train_persisted, y_test_persisted = enc.transform(y_train).persist(), enc.transform(y_test).persist()
    elif task.lower() == 'regression':
        y_train_persisted, y_test_persisted = y_train.persist(), y_test.persist()
    else:
        raise ValueError('Invalid ML task!')

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_scaled, X_test_scaled = sc.transform(X_train).persist(), sc.transform(X_test).persist()

    return X_train_scaled, X_test_scaled, y_train_persisted, y_test_persisted


def dask_default_xgboost_pipeline(X_train: da.Array, X_test: da.Array, y_train: da.Array, y_test: da.Array,
                                  task: str='classification') -> tuple[float, float, da.Array]:

    if task.lower() == 'classification':
        bst_cv = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        bst_eval = xgb.dask.DaskXGBClassifier(random_state=42, eval_metric='logloss')
        regression = False
    elif task.lower() == 'regression':
        bst_cv = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        bst_eval = xgb.dask.DaskXGBRegressor(random_state=42, eval_metric='rmse')
        regression = True
    else:
        raise ValueError('Invalid ML task!')

    cv_scores = dask_cv(bst_cv, X_train, y_train, regression=regression)
    mean_cv_score, std_cv_score = np.mean(cv_scores), np.var(cv_scores)

    # might need to setup client with bst.client
    bst_eval.fit(X_train, y_train)
    prediction = bst_eval.predict(X_test)
    if task.lower() == 'classification':
        eval_score = accuracy_score(y_test, prediction)
    else:
        eval_score = r2_score(y_test, prediction)

    return mean_cv_score, std_cv_score, eval_score


def dask_hpo_sgd_pipeline(param_dist, X_train, X_test, y_train, y_test):
    # Note: XGBoost's Dask implementation does not work with Dask framework HPO methods, so we use the regular one.
    cv_sgd = SGDClassifier(random_state=42, n_jobs=-1)
    cv_search_sgd = RandomizedSearchCV(cv_sgd, param_dist, random_state=42, scoring='accuracy', n_iter=100, cv=2)
    cv_scores = dask_cv(cv_search_sgd, X_train, y_train, cv_splits=5, hpo=True)
    mean_cv_score, std_cv_score = np.mean(cv_scores), np.var(cv_scores)

    sgd = SGDClassifier(random_state=42, n_jobs=-1)
    search_sgd = RandomizedSearchCV(sgd, param_dist, random_state=42, scoring='accuracy', n_iter=100, cv=2)
    search_sgd.fit(X_train, y_train)
    prediction = search_sgd.best_estimator_.predict(X_test.compute())
    eval_score = accuracy_score(y_test.compute(), prediction)

    return mean_cv_score, std_cv_score, eval_score


# TODO: split filepaths into feature path and label path
def dask_pipeline(filepaths: tuple[str, str], task: str='classification',
                  param_dist: dict = None) -> tuple[float, float, float]:
    print(task)
    print("Start")

    print("Load")
    dask_feature_matrix, dask_label_vector = dask_load_data(filepaths)
    dask_feature_matrix = dask_feature_matrix.persist()

    print("Preprocessing")
    preprocessed_feature_matrix = feature_preprocessing(dask_feature_matrix)
    del dask_feature_matrix
    gc.collect()

    print("Convert df to array")
    # For proper splitting into train/test sets, the chunk sizes must align for the two datasets.
    preprocessed_feature_array, label_array = df_to_array(preprocessed_feature_matrix, dask_label_vector)
    del dask_label_vector
    gc.collect()

    print("Train test split")
    X_train, X_test, y_train, y_test = pre_ml_processing(preprocessed_feature_array, label_array, task)
    del preprocessed_feature_array, label_array
    gc.collect()

    print("ML")
    if param_dist:
        print("SGD")
        mean_cv_score, std_cv_score, eval_score = dask_hpo_sgd_pipeline(param_dist, X_train, X_test, y_train, y_test)
    else:
        print("XGBoost")
        mean_cv_score, std_cv_score, eval_score = dask_default_xgboost_pipeline(X_train, X_test, y_train, y_test, task)

    return mean_cv_score, std_cv_score, eval_score
