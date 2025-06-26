import pandas as pd
import matplotlib.pyplot as plt
# This script draw the scatter plot

df = pd.read_csv('../dataset/Location1.csv')

# select the variability from dataset
x_vars = ['windspeed_10m', 'windspeed_100m', 'winddirection_10m', 'winddirection_100m']
y_var = 'Power'
z_vars = ['temperature_2m', 'relativehumidity_2m']

fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
axs = axs.ravel()

colors = ['blue', 'red']

for i, (x, z) in enumerate(zip(x_vars, z_vars * 2)):
    ax = axs[i]
    scatter = ax.scatter(df[x], df[y_var], df[z], c=df[y_var], cmap='coolwarm', alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y_var)
    ax.set_zlabel(z)
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
