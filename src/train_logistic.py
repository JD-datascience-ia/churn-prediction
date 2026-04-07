import pandas as pd

from src.models import logistic_pipeline, logistic_gridsearch
from src.evaluate import run_cross_validation, evaluate_model


def main():

    X_train = pd.read_csv("../data/X_train.csv")
    X_test = pd.read_csv("../data/X_test.csv")
    y_train = pd.read_csv("../data/y_train.csv").squeeze()
    y_test = pd.read_csv("../data/y_test.csv").squeeze()

    pipeline = logistic_pipeline()

    cv_results = run_cross_validation(pipeline, X_train, y_train, cv=5)

    print("=== Cross-validation results ===")
    for metric_name, values in cv_results.items():
        if metric_name.startswith("test_"):
            print(f"{metric_name}: {values.mean():.4f}")

    grid = logistic_gridsearch(cv=5, scoring="recall", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("\n=== Best parameters ===")
    print(grid.best_params_)

    print("\n=== Best CV score ===")
    print(grid.best_score_)


    final_results = evaluate_model(grid.best_estimator_, X_test, y_test)

    print("\n=== Test results ===")
    print(f"Accuracy : {final_results['accuracy']:.4f}")
    print(f"Precision : {final_results['precision']:.4f}")
    print(f"Recall : {final_results['recall']:.4f}")
    print(f"F1-score : {final_results['f1_score']:.4f}")

    print("\n=== Classification report ===")
    print(final_results["classification_report"])



if __name__ == "__main__":
    main()

