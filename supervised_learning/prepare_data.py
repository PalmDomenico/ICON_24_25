import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path, lookback=50, future_shift=5):
    df = pd.read_csv(path)
    df = create_lookback_features(df, target_column="Power", lookback=lookback, future_shift=future_shift)
    X = df.drop(columns=["Power", "Time"], errors="ignore")
    y = df["Power"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    x_train, x_test = normalize_data(x_train, x_test)
    return x_train, x_test, y_train, y_test


def normalize_data(x_train, x_test):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def create_lookback_features(df, target_column, lookback, future_shift=1):
    for i in range(1, lookback + 1):
        df[f"lag_{i}"] = df[target_column].shift(i)
    df[target_column] = df[target_column].shift(-future_shift)
    df.dropna(inplace=True)
    return df


def create_dataset(dataset, lookback=50, future_shift=5):
    X, Y = [], []
    # dataset è un numpy array (samples, features + target)
    for i in range(lookback, len(dataset) - future_shift + 1):
        features = dataset[i - lookback:i, :-1]
        target = dataset[i + future_shift - 1, -1]
        X.append(features)
        Y.append(target)

    return np.array(X), np.array(Y)


def prepare_data_scaled(dataset, train_size, lookback=50, future_shift=5):

    if "Time" in dataset.columns:
        data = dataset.drop(columns=["Time"])
    else:
        data = dataset

    values = data.values

    # 2. Split Train/Test sui dati
    train_data = values[:train_size, :]
    test_data = values[train_size:, :]

    # 3. Normalization
    train_X = train_data[:, :-1]
    train_y = train_data[:, -1].reshape(-1, 1)
    test_X = test_data[:, :-1]
    test_y = test_data[:, -1].reshape(-1, 1)

    # Init scaler
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    # Fit solo su Train
    scaler_x.fit(train_X)
    scaler_y.fit(train_y)

    # Transform on Train and Test
    train_X_scaled = scaler_x.transform(train_X)
    train_y_scaled = scaler_y.transform(train_y)
    test_X_scaled = scaler_x.transform(test_X)
    test_y_scaled = scaler_y.transform(test_y)
    train_scaled_combined = np.hstack((train_X_scaled, train_y_scaled))
    test_scaled_combined = np.hstack((test_X_scaled, test_y_scaled))

    # Sliding Window
    x_train, y_train = create_dataset(train_scaled_combined, lookback, future_shift)
    x_test, y_test = create_dataset(test_scaled_combined, lookback, future_shift)

    return x_train, y_train, x_test, y_test, scaler_x, scaler_y
