import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def scatter_plot(df, save_path):
    x_vars = ['windspeed_10m', 'windspeed_100m', 'winddirection_10m', 'winddirection_100m']
    y_var = 'Power'
    z_vars = ['temperature_2m', 'relativehumidity_2m']
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), subplot_kw={'projection': '3d'})
    axs = axs.ravel()

    for i, (x, z) in enumerate(zip(x_vars, z_vars * 2)):
        ax = axs[i]
        scatter = ax.scatter(df[x], df[y_var], df[z], c=df[y_var], cmap='coolwarm', alpha=0.6)
        ax.set_xlabel(x, labelpad=10, fontweight='bold')
        ax.set_ylabel(y_var, labelpad=10, fontweight='bold')
        ax.set_zlabel(z, labelpad=10, fontweight='bold')
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.15)
        ax.view_init(elev=25, azim=-40)
    plt.subplots_adjust(
        left=0.15,
        right=0.90,
        bottom=0.15,
        top=0.92,
        wspace=0.35,
        hspace=0.35
    )
    plt.savefig(save_path, dpi=400)
    plt.close()


def correlation_heatmap(df, save_path):
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=1,
        square=True,
        annot_kws={"size": 10}
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    folder = "graphs"
    if not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.read_csv('../dataset/join_datasets.csv')
    scatter_plot(df, os.path.join(folder, 'scatter_plot.png'))
    correlation_heatmap(df, os.path.join(folder, 'correlation_heatmap.png'))


if __name__ == '__main__':
    main()
