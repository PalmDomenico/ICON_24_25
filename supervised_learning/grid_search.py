from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from supervised_learning.prepare_data import load_data


def alpha_parameters(model_cv):
    # This graph displays the analyzed alpha parameters
    results = model_cv.cv_results_
    # This get alpha values and relative MSE error
    alphas = results['param_alpha'].data
    mse_values = -results['mean_test_score']

    # Draw graph
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, mse_values, marker='o', linestyle='-', color='b')
    plt.xscale('log')  # Usa scala logaritmica per l'asse x
    plt.xlabel('Valore di alpha')
    plt.ylabel('Errore quadratico medio (MSE)')
    plt.grid(True)
    plt.show()

    print("Miglior valore di alpha:", model_cv.best_params_['alpha'])


def alpha_l1_parameters(elastic_cv, x_test, y_test, param_grid):
    # This graph displays the analyzed alpha and l1 parameters
    top_alpha = elastic_cv.best_params_['alpha']
    top_l1_ratio = elastic_cv.best_params_['l1_ratio']

    # Value on test set
    y_pred = elastic_cv.best_estimator_.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE sul test set: {mse}')
    # Grafico dell'errore rispetto ad alpha
    alphas = param_grid['alpha']
    l1_ratios = param_grid['l1_ratio']
    mse_scores = -elastic_cv.cv_results_['mean_test_score'].reshape(len(l1_ratios), len(alphas))
    # draw graph
    plt.figure(figsize=(8, 5))
    for i, l1 in enumerate(l1_ratios):
        plt.plot(alphas, mse_scores[i], marker='o', linestyle='--', label=f'l1_ratio={l1:.1f}')
    plt.axvline(x=top_alpha, color='r', linestyle='-', label=f'Miglior alpha: {top_alpha}')
    plt.xscale('log')  # Scala logaritmica per alpha
    plt.xlabel('Valore di alpha')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Andamento del MSE in funzione di alpha e l1_ratio')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Function for optimizing hyperparameters of different algorithms
    x_train, x_test, y_train, y_test = load_data('../dataset/Location1.csv')

    param_grid = {
        'alpha': np.logspace(-6, -2, 4),
    }

    # Ridge Regression
    ridge = Ridge()
    model_cv = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
    model_cv.fit(x_train, y_train)
    alpha_parameters(model_cv)

    # Lasso Regression
    lasso = Lasso()
    model_cv = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
    model_cv.fit(x_train, y_train)
    alpha_parameters(model_cv)

    # ElasticNet Regression
    elastic_net = ElasticNet()
    param_grid = {
        'alpha': np.logspace(-6, 6, 13),  # Valori da 10^-6 a 10^6 su scala log
        'l1_ratio': np.linspace(0, 1, 11)  # Valori tra 0 (Ridge) e 1 (Lasso)
    }
    elastic_cv = GridSearchCV(elastic_net, param_grid, scoring='neg_mean_squared_error', cv=5)
    elastic_cv.fit(x_train, y_train)
    alpha_l1_parameters(elastic_cv, x_test, y_test, param_grid)

    param_grid = {
        'max_depth': [3, 5, 10, None],  # Profondità massima dell'albero
        'min_samples_split': [2, 5, 10],  # Numero minimo di campioni per dividere un nodo
    }
    # Modello ElasticNet
    elastic_net = DecisionTreeRegressor()

    # Grid Search con validazione incrociata
    elastic_cv = GridSearchCV(elastic_net, param_grid, verbose=3, scoring='neg_mean_squared_error', cv=5)
    elastic_cv.fit(x_train, y_train)

    top_max_depth = elastic_cv.best_params_['max_depth']
    top_min_samples_split = elastic_cv.best_params_['min_samples_split']
    print(f'Miglior valore di max_depth: {top_max_depth}')
    print(f'Miglior valore di min_samples_split: {top_min_samples_split}')


if __name__ == '__main__':
    main()
