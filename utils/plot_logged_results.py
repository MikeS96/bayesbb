from numpy import float64
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification

# Reading scared logs to pandas dataframe

ens_dm_df = pd.read_json('sacred_files/71/metrics.json')
# have to comment out lines below because sacred won't save logs
# drp_dm_df = pd.read_json('sacred_files/70/metrics.json')
drp_dm_df = pd.read_csv('assets/baselines/data/drp_dirty_mnist_test accuracy.csv')
# det_dm_df = pd.read_json('sacred_files/69/metrics.json')
det_dm_df = pd.read_csv('assets/baselines/data/det_dirty_mnist_test accuracy.csv')


# We will take the values we want and put them in this dataframe 

final_plot_df = pd.DataFrame()
final_plot_df['Ensemble'] = ens_dm_df['test accuracy']['values']
# reading from csv files is different because sacred won't save logs :(
final_plot_df['Dropout'] = drp_dm_df['0.7522141933441162']
final_plot_df['Deterministic'] = det_dm_df['0.7595000863075256']

# this adds some smoothing to the plot
final_plot_df = final_plot_df[final_plot_df.index % 2 == 0] 



# We can even edit df values, such as correcting for negative entropy
# final_plot_df['Ensemble Mnist Test Entropy'] = ens_dm_df['test entropy']['values']
# final_plot_df['Ensemble Mnist Test Entropy'] *= -1

# Naming the axes
final_plot_df.index.name = "Epochs"
final_plot_df.columns.name = "Test Accuracy"

# Add a title and/or other properties to the graph
plot_title = 'Baseline Ensemble Mnist Results'
final_plot = px.line(
    final_plot_df, title=plot_title,
    width=500, height=350  , line_shape='spline', 
)

# Center Title
final_plot.update_layout(
    title={
        'text': plot_title,
        'y':0.8,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'
        })

# Other possible edits you can make to the graph
# fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
# final_plot.update_yaxes(range=[0.7, 0.9], constrain='domain')
# final_plot.update_xaxes(range=[0, 100], dtick = 5, constrain='domain')

final_plot.show()

