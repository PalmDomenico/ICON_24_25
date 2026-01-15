import os
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from KB.prolog import write_facts, query_kb_features
from supervised_learning.evalue_model import evaluate_model, metrics_graph
from supervised_learning.prepare_data import load_data
from supervised_learning.neural_network import train_neural_network
import pandas as pd
from unsupervised_learning.cluster import calculate_cluster


def models_train(dataset_path, save_path="models7", load_existing=True):
    lookback = 50
    future_shift = 0
    # load data
    x_train, x_test, y_train, y_test = load_data(dataset_path, lookback=lookback, future_shift=future_shift)
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
        "LinearSVR": LinearSVR(max_iter=10000)
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


def join_datasets(folder_path, csv_name, sub_folder_name="original_datasets"):
    locations = ["Location1.csv", "Location2.csv", "Location3.csv", "Location4.csv"]
    df_list = []

    for loc in locations:
        file_path = os.path.join(folder_path, sub_folder_name, loc)
        df = pd.read_csv(str(file_path))
        df['LocationID'] = loc.split('.')[0]
        df_list.append(df)

    combined_dataset = pd.concat(df_list, ignore_index=True)
    combined_dataset.to_csv(os.path.join(folder_path, str(csv_name)), index=False)


def main():
    prep_data = True
    if prep_data:
        folder_path = "dataset"
        csv_name_join = "join_datasets.csv"
        csv_name_clustered = "clustered_dataset.csv"
        csv_name_extended = "extended_dataset.csv"
        kb_rules_file = "rules.pl"
        kb_facts_file = "facts.pl"

        # 1. Join data
        csv_path_join = os.path.join(folder_path, csv_name_join)
        if not os.path.exists(csv_path_join):
            join_datasets(folder_path, csv_name_join)
        df_raw = pd.read_csv(csv_path_join)

        # 2. Clustering
        scaler_temp = MinMaxScaler()
        numeric_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns
        df_temp_norm = df_raw.copy()
        df_temp_norm[numeric_cols] = scaler_temp.fit_transform(df_raw[numeric_cols])
        labels = calculate_cluster(df_temp_norm)
        df_raw['Cluster'] = labels
        df_raw.to_csv(os.path.join(folder_path, csv_name_clustered), index=False)

        # 3. Interaction with Knowledge Base
        write_facts(df_raw=df_raw, df_cluster=df_raw, filename=kb_facts_file)
        df_extended = query_kb_features(
            df_target=df_raw,
            rules_file=kb_rules_file,
            facts_file=kb_facts_file
        )
        csv_path_extended = os.path.join(folder_path, csv_name_extended)
        df_extended.to_csv(csv_path_extended, index=False)


if __name__ == "__main__":
    main()
