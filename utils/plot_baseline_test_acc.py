from numpy import float64
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification

### Reading logs to dataframes ###

### Clean Mnist ###
ens_cm_df = pd.read_csv('assets/baselines/data/ens_clean_mnist_test_acc.csv')
drp_cm_df = pd.read_csv('assets/baselines/data/drp_clean_mnist_test_acc.csv')
det_cm_df = pd.read_csv('assets/baselines/data/det_clean_mnist_test_acc.csv')

# We will take the values we want and put them in this dataframe
cm_plot_df = pd.DataFrame()
cm_plot_df['Ensemble'] = ens_cm_df.iloc[:, 2]
cm_plot_df['Dropout'] = drp_cm_df.iloc[:, 2]
cm_plot_df['Deterministic'] = det_cm_df.iloc[:, 2]

# We can even edit df values, such as correcting for negative entropy
# final_plot_df['Ensemble Mnist Test Entropy'] *= -1

### Dirty Mnist ###
ens_dm_df = pd.read_csv('assets/baselines/data/ens_dirty_mnist_test_acc.csv')
drp_dm_df = pd.read_csv('assets/baselines/data/drp_dirty_mnist_test_acc.csv')
det_dm_df = pd.read_csv('assets/baselines/data/det_dirty_mnist_test_acc.csv')


# We will take the values we want and put them in this dataframe
dm_plot_df = pd.DataFrame()
dm_plot_df['Ensemble'] = ens_dm_df.iloc[:, 2]
dm_plot_df['Dropout'] = drp_dm_df.iloc[:, 2]
dm_plot_df['Deterministic'] = det_dm_df.iloc[:, 2]

df_to_plot = {"clean_mnist_baslines": [
    "Clean Mnist", cm_plot_df], "dirty_mnist_baslines": ["Dirty Mnist", dm_plot_df]}

# Plot both dataframes!
for key, stuff in df_to_plot.items():

    curr_label = stuff[0]
    df = stuff[1]
    # this adds some smoothing to the plot
    df = df[df.index % 2 == 0]

    # Naming the axes
    df.index.name = "Epochs"
    df.columns.name = f"Baseline {curr_label} Test Accuracy"

    # Add a title and/or other properties to the graph
    plot_title = ''
    final_plot = px.line(
        df, title=plot_title,
        width=400, height=250,
        line_shape='spline',
        template='simple_white',
        labels={'value': 'Test Accuracy'}
    )

    # Modify Apperance of the plot
    final_plot.update_layout(
        title={
            'text': plot_title,
            'y': 0.8,
            'x': 0.45,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1.0
        )
    )

    # # Remove grid and background
    # final_plot.update_layout({
    # 'plot_bgcolor': 'rgba(255, 255, 255, 255)',
    # 'paper_bgcolor': 'rgba(255, 255, 255, 255)',
    # 'mirror': 'True'
    # })

    # Other possible edits you can make to the graph
    # fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
    # final_plot.update_yaxes(range=[0.7, 0.9], constrain='domain')
    # final_plot.update_xaxes(range=[0, 100], dtick = 5, constrain='domain')

    final_plot.show()
    final_plot.write_image(f"assets/baselines/figs/{key}.svg")
