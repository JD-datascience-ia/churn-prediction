from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

def randomforest_pipeline():
    pass

def xgboost_pipeline():
    pass




