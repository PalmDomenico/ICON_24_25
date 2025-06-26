import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path, lookback=50, future_shift=5):
    df = pd.read_csv(path)

    # Crea le feature con lookback e spostamento futuro
    df = create_lookback_features(df, target_column="Power", lookback=lookback, future_shift=future_shift)

    # Prepara X e y
    X = df.drop(columns=["Power", "Time"], errors="ignore")
    y = df["Power"]

    # Split e normalizzazione
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Sposta il target verso il passato per simulare la previsione futura
    df[target_column] = df[target_column].shift(-future_shift)

    df.dropna(inplace=True)
    return df


def create_dataset(dataset, lookback=50, future_shift=5):
    # Lookback 3D dataset
    X, Y = [], []
    for i in range(lookback, len(dataset) - future_shift + 1):
        a = np.concatenate((dataset[i - lookback:i, 0:8], dataset[i - lookback:i, 9:16]), axis=1)
        X.append(a)

        # Output = valore target a t + future_shift
        Y.append(dataset[i + future_shift - 1, 8])

    return np.array(X), np.array(Y)


def prepare_data_scaled(dataset, train_size, lookback=50, future_shift=5):
    # Rimuovi la colonna Time
    data = dataset.drop(columns=["Time"])

    # Separa le feature dalla variabile target (ultima colonna)
    values = data.values
    X = values[:, :-1]  # tutte le colonne tranne l'ultima
    y = values[:, -1].reshape(-1, 1)  # ultima colonna come target

    # Normalizza separatamente
    scaler_x = MinMaxScaler(feature_range=(-1, 0))
    X_scaled = scaler_x.fit_transform(X)

    scaler_y = MinMaxScaler(feature_range=(-1, 0))
    y_scaled = scaler_y.fit_transform(y)

    # Ricombina X e y scalati
    scaled_data = np.hstack((X_scaled, y_scaled))


    # Split train/test
    train = scaled_data[:train_size - lookback, :]
    test = scaled_data[train_size - lookback:, :]

    x_train, y_train = create_dataset(train, lookback=50, future_shift=5)
    x_test, y_test = create_dataset(test, lookback=50, future_shift=5)

    # reshape input in [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2], x_test.shape[1]))

    return x_train, y_train, x_test, y_test, scaler_x, scaler_y
