import os
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from supervised_learning.evalue_model import evaluate_model, metrics_graph
from supervised_learning.prepare_data import load_data
from supervised_learning.neural_network import train_neural_network
import pandas as pd
from KB.prolog import df_to_prolog_facts, append_facts, add_rules, integrate_logical_features
from unsupervised_learning.cluster import calculate_cluster


def models_train(dataset_path, save_path="models7", load_existing=True):
    lookback = 50
    future_shift = 0
    # load data
    x_train, x_test, y_train, y_test = load_data(dataset_path,lookback=lookback, future_shift=future_shift)
    os.makedirs(save_path, exist_ok=True)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1e-06),
        "Lasso Regression": Lasso(alpha=1e-06),
        "ElasticNet Regression": ElasticNet(alpha=1e-06, l1_ratio=0.7),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, verbose=2),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, min_samples_split=10),
        "Support Vector Regressor": SVR(C=1.0, epsilon=0.1),
        "LinearSVR":LinearSVR(max_iter=10000)
    }

    results = {}
    # train all models and evalue
    for name, model in models.items():
        model_path = os.path.join(save_path, f"{name.replace(' ', '_')}.pkl")

        if load_existing and os.path.exists(model_path):
            print(f"Loading existing model: {name}")
            model = joblib.load(model_path)
        else:
            print(f"Training model: {name}")
            model.fit(x_train, y_train)
            joblib.dump(model, model_path)

        y_pred = model.predict(x_test)
        results[name] = evaluate_model(y_test, y_pred)
    # train neural network model
    results["LSTM"] = train_neural_network(dataset_path, lookback=lookback, future_shift=future_shift)
    metrics_graph(results)
    return results


def create_KB(original_dataset_path, extended_dataset_path, prolog_file_path):
    original_dataset = pd.read_csv(original_dataset_path)
    df_to_prolog_facts(original_dataset, prolog_file_path)  # convert dataset in prolog format
    labels = calculate_cluster(original_dataset, extended_dataset_path)
    append_facts(prolog_file_path, labels)  # add new facts to prolog
    add_rules(prolog_file_path)
    integrate_logical_features(extended_dataset_path, prolog_file_path)  # extend dataset with query on prolog


def main():
    original_dataset_path = os.path.join("dataset", "Location1.csv")
    extended_dataset_path = os.path.join("dataset", "extended_dataset4.csv")
    prolog_file_path = "knowledge_base3.pl"
    create_KB(original_dataset_path, extended_dataset_path, prolog_file_path)

    models_train(extended_dataset_path)


if __name__ == "__main__":
    main()
