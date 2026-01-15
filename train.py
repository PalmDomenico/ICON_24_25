import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from supervised_learning.evalue_model import evaluate_model
from supervised_learning.prepare_data import create_lookback_features


def load_and_process_safe(path, lookback=50, future_shift=5):
    df = pd.read_csv(path)
    processed_chunks = []
    locations = df['LocationID'].unique()
    for loc in locations:
        df_loc = df[df['LocationID'] == loc].copy()
        if 'Time' in df_loc.columns:
            df_loc['Time'] = pd.to_datetime(df_loc['Time'])
            df_loc = df_loc.sort_values('Time')
        df_loc = create_lookback_features(df_loc, target_column="Power", lookback=lookback, future_shift=future_shift)
        processed_chunks.append(df_loc)
    df_final = pd.concat(processed_chunks, ignore_index=True)

    cols_to_drop = ["Time", "LocationID"]
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], errors="ignore")
    y = df_final["Power"].values.reshape(-1, 1)
    X = df_final.drop(columns=["Power"]).values
    return X, y


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_eval_kfold(X, y, models_dict, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)

    metrics_schema = ['mse', 'rmse', 'mae', 'std_abs_err', 'train_time', 'pred_time']
    results = {name: {m: [] for m in metrics_schema} for name in models_dict.keys()}
    results['LSTM'] = {m: [] for m in metrics_schema}

    print(f"\n--- Avvio K-Fold Cross Validation (K={n_splits}) ---")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):  # Kfold
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler_x.fit_transform(X_train_raw)
        X_val = scaler_x.transform(X_val_raw)
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        y_train = scaler_y.fit_transform(y_train_raw)

        for name, model in models_dict.items():
            print(f"\n--- {name} ---")
            from sklearn.base import clone
            clf = clone(model)
            clf.fit(X_train, y_train.ravel())
            y_pred_scaled = clf.predict(X_val)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            mse, rmse, mae, std_abs = evaluate_model(y_val_raw.ravel(), y_pred.ravel())

            results[name]['mse'].append(mse)
            results[name]['rmse'].append(rmse)
            results[name]['mae'].append(mae)
            results[name]['std_abs_err'].append(std_abs)

        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        lstm = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=3)

        lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=32,
                 validation_data=(X_val_lstm, scaler_y.transform(y_val_raw)),
                 callbacks=[es], verbose=0)

        y_pred_scaled = lstm.predict(X_val_lstm, verbose=0)

        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        mse, rmse, mae, std_abs = evaluate_model(y_val_raw.ravel(), y_pred.ravel())
        results['LSTM']['mse'].append(mse)
        results['LSTM']['rmse'].append(rmse)
        results['LSTM']['mae'].append(mae)
        results['LSTM']['std_abs_err'].append(std_abs)

    return results


def print_full_report(results, n_splits):
    headers = f"{'Modello':<25} | {'RMSE':<12} | {'MAE':<12} | {'MSE':<12} | {'Std. Err':<12} | {'Train Time (s)':<15}"
    print(headers)

    # Ordinamento per RMSE crescente (migliore in alto)
    sorted_models = sorted(results.keys(), key=lambda x: np.mean(results[x]['rmse']))

    for name in sorted_models:
        metrics = results[name]
        rmse_str = f"{np.mean(metrics['rmse']):.4f} (±{np.std(metrics['rmse']):.3f})"
        mae_str = f"{np.mean(metrics['mae']):.4f}"
        mse_str = f"{np.mean(metrics['mse']):.4f}"
        std_str = f"{np.mean(metrics['std_abs_err']):.4f}"
        time_str = f"{np.mean(metrics['train_time']):.3f}"

        print(f"{name:<25} | {rmse_str:<12} | {mae_str:<12} | {mse_str:<12} | {std_str:<12}")

    print("=" * 120 + "\n")

    # Generazione Grafico Comparativo
    plot_comparison(results, n_splits)


def plot_comparison(results, n_splits):
    models = list(results.keys())
    rmse_means = [np.mean(results[m]['rmse']) for m in models]

    plt.figure(figsize=(12, 6))
    plt.bar(models, rmse_means, color='skyblue', edgecolor='black')
    plt.title(f'Confronto RMSE Medio ({n_splits}-Fold CV)')
    plt.ylabel('RMSE (minore è meglio)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')


def train():
    lookback = 50
    future_shift = 0
    n_splits = 5

    # 1. Load and preprocessing data
    dataset_file = os.path.join("dataset", "extended_dataset.csv")
    X, y = load_and_process_safe(dataset_file, lookback=lookback, future_shift=future_shift)

    # 2. Definition model
    sklearn_models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1e-06),
        "Lasso": Lasso(alpha=1e-06),
        "ElasticNet": ElasticNet(alpha=1e-06, l1_ratio=0.7),
        "Random Forest": RandomForestRegressor(n_estimators=50, verbose=0, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42),
        "SVR": SVR(C=1.0, epsilon=0.1),
        "LinearSVR": LinearSVR(max_iter=10000, random_state=42, dual='auto')
    }

    # 3. Training and evaluation
    results = train_eval_kfold(X, y, sklearn_models)

    # 4. Report Finale
    print_full_report(results, n_splits)

    joblib.dump(results, "cv_results.pkl")


if __name__ == "__main__":
    train()
