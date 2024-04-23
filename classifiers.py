import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from itertools import product

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sns.set_style('whitegrid')


def lsvc_classifier(X_train, y_train, X_test, y_test):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, y_train)

    train_score = linear_svc.score(X_train, y_train)
    test_score = linear_svc.score(X_test, y_test)
    
    b_pred_svc = linear_svc.decision_function(X_test)
    auc_roc_svc_ = roc_auc_score(y_test, b_pred_svc)
    auc_pr_svc_ = average_precision_score(y_test, b_pred_svc)
    
    return linear_svc, {
        'train_acc': train_score, 'test_acc': test_score, 
        'auc_roc_test': auc_roc_svc_, 'auc_pr_test': auc_pr_svc_
    }


def decision_tree_classifier(
    X_train, y_train, X_test, y_test, 
    depth_grid=range(3, 16), samples_leaf_grid=range(1, 5), random_forest=False
):
    models = {}
    accuracy = {part: np.zeros((len(depth_grid), len(samples_leaf_grid))) for part in ['train', 'test']}

    for i, depth in tqdm(enumerate(depth_grid), total=len(depth_grid), leave=False):
        for j, samples_leaf in enumerate(samples_leaf_grid):
            if random_forest:
                model = RandomForestClassifier(
                    max_depth = depth, 
                    min_samples_leaf = samples_leaf
                ).fit(X_train, y_train)
            else:
                model = DecisionTreeClassifier(
                    max_depth = depth, 
                    min_samples_leaf = samples_leaf
                ).fit(X_train, y_train)
                
            pred_train = model.predict(X_train)
            pred = model.predict(X_test)
            accuracy['train'][i, j] = accuracy_score(y_train, pred_train) 
            accuracy['test'][i, j] = accuracy_score(y_test, pred)
            models[(depth, samples_leaf)] = model
            
    for part in accuracy:
        accuracy[part] = pd.DataFrame(accuracy[part])
        accuracy[part].columns = samples_leaf_grid
        accuracy[part].index = depth_grid
        
    return models, accuracy

def full_pipeline(df):
    df_train = df.query('part == \'train\'')
    df_test = df.query('part == \'test\'')
        
    X_train = df_train.iloc[:, :-3]
    y_train = df_train.text_type == 'lit'

    X_test = df_test.iloc[:, :-3]
    y_test = df_test.text_type == 'lit'
    
    res_lsvc = lsvc_classifier(X_train.to_numpy(), y_train, X_test.to_numpy(), y_test)
    res_dt = decision_tree_classifier(X_train.to_numpy(), y_train, X_test.to_numpy(), y_test)
    res_rf = decision_tree_classifier(
        X_train.to_numpy(), y_train, X_test.to_numpy(), y_test, 
        samples_leaf_grid=range(1, 11),
        random_forest=True
    )
    
    res_clf = {}

    res_rf_test = res_rf[1]['test']
    res_rf_train = res_rf[1]['train']
    best_idx = res_rf_test.to_numpy().argmax()
    d, s = list(product(res_rf_test.index, res_rf_test.columns))[best_idx]
    params = {'depth': d, 'samples_leaf': s}
    train_acc = res_rf_train.to_numpy().flatten()[best_idx]
    test_acc = res_rf_test.to_numpy().flatten()[best_idx]
    res_clf['random_forest'] = {'train_acc': train_acc, 'test_acc': test_acc, 'params': params, 'model': res_rf[0][(d, s)]}

    res_dt_test = res_dt[1]['test']
    res_dt_train = res_dt[1]['train']
    best_idx = res_dt_test.to_numpy().argmax()
    d, s = list(product(res_dt_test.index, res_dt_test.columns))[best_idx]
    params = {'depth': d, 'samples_leaf': s}
    train_acc = res_dt_train.to_numpy().flatten()[best_idx]
    test_acc = res_dt_test.to_numpy().flatten()[best_idx]
    res_clf['decision_tree'] = {'train_acc': train_acc, 'test_acc': test_acc, 'params': params, 'model': res_rf[0][(d, s)]}
    
    res_clf['lsvc'] = {'model': res_lsvc[0]} | res_lsvc[1]
    
    return res_clf
