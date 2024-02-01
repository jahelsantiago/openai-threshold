import glob
from graph_for_metric_vs_threshold import plot_metrics
from histogram_companies_vs_cosine_similarity import plot_histogram


# Get all CSV files in the folder
csv_files = glob.glob('*.csv')

for file in csv_files:
    plot_metrics(file)
    plot_histogram(file)
