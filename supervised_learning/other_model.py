import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

# support vector regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# used for testing
def svr_model(df, input_data):
    df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'],
            axis=1,
            inplace=True)

    y = df['co2_emissions']
    df.drop('co2_emissions', axis=1, inplace=True)
    # print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)

    regr = LinearSVR(max_iter=10000)

    regr.fit(X_train, y_train)

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = int(regr.predict(input_data_reshaped))

    # uncomment to print the test and train accuracies
    # print("svr train accuracy", regr.score(X_train, y_train))
    # print("svr TEST accuracy", regr.score(X_test, y_test))

    return prediction


# used for testing
def decision_tree_model(df, input_data):
    df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'],
            axis=1,
            inplace=True)

    y = df['co2_emissions']
    df.drop('co2_emissions', axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)

    decision_tree = DecisionTreeRegressor()

    decision_tree.fit(X_train, y_train)
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = int(decision_tree.predict(input_data_reshaped))

    # uncomment to print the test and train accuracies
    # print("Decision tree train accuracy", decision_tree.score(X_train, y_train))
    # print("Decision tree TEST accuracy", decision_tree.score(X_test, y_test))

    return prediction


def rfr_model(df, input_data):
    df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'],
            axis=1,
            inplace=True)

    y = df['co2_emissions']
    df.drop('co2_emissions', axis=1, inplace=True)
    # print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)

    regr = RandomForestRegressor(max_depth=7, n_estimators=100,
                                 random_state=False, verbose=False)

    regr.fit(X_train, y_train)

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = int(regr.predict(input_data_reshaped))

    # uncomment to print the test and train accuracies
    # print("rfr train accuracy", regr.score(X_train, y_train))
    # print("rfr TEST accuracy", regr.score(X_test, y_test))

    return prediction
