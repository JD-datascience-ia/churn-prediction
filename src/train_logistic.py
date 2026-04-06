import pandas as pd

from src.models import build_logistic_pipeline, build_logistic_gridsearch
from src.evaluate import run_cross_validation, evaluate_model


def main():

    X_train = pd.read_csv("../data/X_train.csv")
    X_test = pd.read_csv("../data/X_test.csv")
    y_train = pd.read_csv("../data/y_train.csv").squeeze()
    y_test = pd.read_csv("../data/y_test.csv").squeeze()

    pipeline = build_logistic_pipeline()

    