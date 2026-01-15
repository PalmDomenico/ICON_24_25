from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(y_true, y_pred):
    # Metriche base
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Deviazione standard degli errori assoluti
    abs_errors = np.abs(np.array(y_true) - np.array(y_pred))
    std_abs_error = np.std(abs_errors, ddof=1)  # campionaria

    return mse, rmse, mae, std_abs_error


def metrics_graph(results):
    # draw graph metrics
    models = list(results.keys())
    print(models)
    mse_values, rmse_values, mae_values, std_abs_error = zip(*results.values())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [mse_values, rmse_values, mae_values, std_abs_error]
    metrics = [[round(num, 4) for num in sub_list] for sub_list in metrics]
    print(metrics)
    titles = ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)",
              "Mean Absolute Error (MAE)", "Deviazione standard"]

    for i, ax in enumerate(axes.flat):
        ax.bar(models, metrics[i], color=['blue', 'green', 'red', 'purple'])
        ax.set_title(titles[i])
        ax.set_ylabel("Value")
        ax.set_xticklabels(models, rotation=20)

    plt.tight_layout()
    plt.show()

