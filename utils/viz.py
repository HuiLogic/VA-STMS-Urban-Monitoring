import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def setup_viz():
    # Setup visualization style
    sns.set_theme(style="darkgrid")
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['figure.figsize'] = (10, 6)

def save_fig(fig, path, name):
    # Save figure in both PDF and high-res PNG
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{path}/{name}.png", bbox_inches='tight', dpi=300)
    print(f"Saved figure: {name}")

def export_table(df, path, name):
    # Export dataframe to CSV
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/{name}.csv", index=False)