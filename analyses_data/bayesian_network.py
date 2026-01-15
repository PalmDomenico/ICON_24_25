import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import BayesianEstimator


def load_data(file_path):
    """Loads dataset and performs basic cleaning."""
    try:
        df = pd.read_csv(file_path)
        df_clean = df.drop(columns=['Time', 'LocationID']) if 'Time' in df.columns else df
        return df_clean
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None


def preprocess_and_discretize(df, n_bins=5):
    """Discretizes continuous variables into ordinal bins."""
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    data_discrete = discretizer.fit_transform(df)
    df_discrete = pd.DataFrame(data_discrete, columns=df.columns).astype(int).astype(str)
    return df_discrete


def train_bayesian_model(df_discrete, max_iter=1000):
    hc = HillClimbSearch(df_discrete)
    structure = hc.estimate(scoring_method='k2', max_iter=max_iter)
    model = DiscreteBayesianNetwork(structure.edges())
    model.fit(
        df_discrete,
        estimator=BayesianEstimator,
        prior_type='BDeu',
        equivalent_sample_size=10
    )
    return model


def plot_bayesian_graph(model, save_path='bayesian_network_structure.png'):
    plt.figure(figsize=(16, 10))
    G = nx.MultiDiGraph(model.edges())
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color="#ff574c", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif")
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowsize=40,
        arrowstyle='-|>',
        edge_color="black",
        width=2.5,
        connectionstyle="arc3,rad=0.15",
        min_source_margin=20,
        min_target_margin=30
    )
    plt.title("Bayesian Network Structural Dependencies", fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_power_distribution(df_discrete, save_path):
    """Generates a pie chart of the discretized Power levels."""
    plt.figure(figsize=(8, 8))
    counts = df_discrete['Power'].value_counts().sort_index()
    plt.pie(counts, labels=[f"Bin {i}" for i in counts.index], autopct='%1.1f%%',
            colors=sns.color_palette("viridis"))
    plt.title("Power Production Distribution", fontsize=14, fontweight='bold')
    plt.savefig(save_path)
    plt.close()


def run_inference_analysis(model, save_path):
    """Compares Power probability distributions for high vs low wind speed."""
    inference = VariableElimination(model)
    res_high = inference.query(variables=['Power'], evidence={'windspeed_100m': '4'}).values
    res_low = inference.query(variables=['Power'], evidence={'windspeed_100m': '0'}).values

    x = np.arange(5)
    plt.figure(figsize=(10, 6))
    plt.bar(x + 0.2, res_high, 0.4, label='Wind Speed 100m (Bin 4)', color='orange')
    plt.bar(x - 0.2, res_low, 0.4, label='Wind Speed 100m (Bin 0)', color='skyblue')

    plt.xlabel('Production Class (Power)')
    plt.ylabel('Probability')
    plt.title("Inference Analysis: Power Given Wind Speed", fontsize=14, fontweight='bold')
    plt.xticks(x, [f'P({i})' for i in range(5)])
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def generate_3_variable_cpd_image(model, target, p1, p2, save_path):
    inference = VariableElimination(model)
    states_to_show = ['0', '2', '4']
    cpd_data = []

    for s1 in states_to_show:
        for s2 in states_to_show:
            try:
                res = inference.query(variables=[target], evidence={p1: s1, p2: s2}, show_progress=False)
                row = [f"{p1}({s1})", f"{p2}({s2})"]
                row.extend([f"{prob:.4f}" for prob in res.values])
                cpd_data.append(row)
            except:
                continue

    target_states = model.get_cpds(target).state_names[target]
    columns = [f"State {p1}", f"State {p2}"] + [f"P({target}={s})" for s in target_states]
    df_cpd = pd.DataFrame(cpd_data, columns=columns)
    save_table_as_image(df_cpd, save_path, color='#cfe2ff')


def save_table_as_image(df, save_path, color='#cfe2ff'):
    num_cols = len(df.columns)
    fig_width = max(12, num_cols * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, len(df) * 0.8 + 2))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colColours=[color] * len(df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    table.auto_set_column_width(col=list(range(len(df.columns))))

    for (row, col), cell in table.get_celld().items():
        cell.set_height(0.12)
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
        elif col < 2:
            cell.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    folder = "graphs"
    if not os.path.exists(folder): os.makedirs(folder)

    df_raw = load_data('../dataset/join_datasets.csv')
    if df_raw is None: return

    df_discrete = preprocess_and_discretize(df_raw)
    model = train_bayesian_model(df_discrete)

    # Standard Visualizations
    plot_bayesian_graph(model, os.path.join(folder, 'bayesian_network_graph.png'))
    plot_power_distribution(df_discrete, os.path.join(folder, 'energy_distribution.png'))
    run_inference_analysis(model, os.path.join(folder, 'wind_impact_analysis.png'))

    # CPD Generate Table
    generate_3_variable_cpd_image(model, 'Power', 'windspeed_100m', 'temperature_2m',
                                  os.path.join(folder, 'cpd_power_wind_temp.png'))

    # 2. Power vs Wind Speed & Wind Direction (Efficiency based on orientation)
    generate_3_variable_cpd_image(model, 'Power', 'windspeed_100m', 'winddirection_100m',
                                  os.path.join(folder, 'cpd_power_wind_direction.png'))

    # 3. Power vs Wind Speed & Wind Gusts (Impact of turbulence)
    generate_3_variable_cpd_image(model, 'Power', 'windspeed_100m', 'windgusts_10m',
                                  os.path.join(folder, 'cpd_power_wind_gusts.png'))


if __name__ == "__main__":
    main()
