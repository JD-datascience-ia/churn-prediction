from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def logistic_pipeline():

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=42)
    )
    return pipeline

def logistic_param_grid():

    param_grid = {
        "logisticregression__C": [0.01, 0.1, 1, 10],
        "logisticregression__solver": ["lbfgs", "liblinear"],
        "logisticregression__class_weight": [None, "balanced"]
    }
    return param_grid

def logistic_gridsearch(cv=5, scoring="recall", n_jobs=-1):

    pipeline = logistic_pipeline()
    param_grid = logistic_param_grid()

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )
    return grid









def randomforest_model():
    model = RandomForestClassifier(random_state=42)
    
    return model

def randomforest_param_grid():

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 4, 6],
        "max_features": ["sqrt", 0.5],
        "class_weight": [None, "balanced"]
    }
    return param_grid

def randomforest_gridsearch(cv=5, scoring="recall", n_jobs=-1):

    model = randomforest_model()
    param_grid = randomforest_param_grid()

    grid = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        cv = cv,
        scoring = scoring,
        n_jobs = n_jobs
    )
    return grid









def xgboost_model():
    
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
    
    return model

def xgboost_param_grid():


    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "min_child_weight": [1, 3],
        "gamma": [0, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "scale_pos_weight": [1, 3]
    }
    return param_grid

def xgboost_gridsearch(cv=5, scoring="recall", n_jobs=-1):

    model = xgboost_model()
    param_grid = xgboost_param_grid()

    grid = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        cv = cv,
        scoring = scoring,
        n_jobs = n_jobs
    )
    return grid
    




