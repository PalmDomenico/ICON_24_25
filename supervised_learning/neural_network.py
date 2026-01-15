import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd

from supervised_learning.evalue_model import evaluate_model
from supervised_learning.prepare_data import prepare_data_scaled


def build_neural_network(shape):
    # create neural network model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_neural_network(dataset_path, lookback=50, future_shift=5):
    train_model = False
    path_model = "models/neural_network.keras"

    dataset = pd.read_csv(dataset_path)

    train_size = int(0.85 * len(dataset))

    # Preparazione dei dati (restituisce scaler_y separato)
    x_train, y_train, x_test, y_test, scaler_x, scaler_y = prepare_data_scaled(dataset, train_size, lookback=lookback,
                                                                               future_shift=future_shift)

    if train_model:
        model = build_neural_network((x_train.shape[1], x_train.shape[2]))
        model.fit(x_train, y_train, batch_size=1, epochs=3, validation_data=(x_test, y_test))
        model.save(path_model)
    else:
        model = keras.models.load_model(path_model)

    model.summary()

    # Predizione
    test_predict = model.predict(x_test)

    # Inverse transform corretta (assumendo 1 sola feature target)
    test_predict = scaler_y.inverse_transform(test_predict.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Valutazione
    statistic = evaluate_model(y_test.ravel(), test_predict.ravel())
    results = {"LSTM": statistic}

    return statistic
