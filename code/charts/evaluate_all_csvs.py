import glob
from graph_for_metric_vs_threshold import plot_metrics
from histogram_companies_vs_cosine_similarity_csv import plot_histogram
from extract_name import extract_name


# Get all CSV files in the folder

csv_files = glob.glob(
    '/Users/esteban/Documents/VIQ/VIQ-770/openai-threshold/csvs/*.csv')

for file in csv_files:
    name = extract_name(file)
    print(name)
    print(file)
    if name.startswith('eval_10K'):
        plot_metrics(file)
        plot_histogram(file)
    else:
        plot_histogram(file)
        plot_metrics(file)

file = '/Users/esteban/Documents/VIQ/VIQ-770/openai-threshold/csvs/results-real-state-10k.csv' # noqa
plot_metrics(file)
plot_histogram(file)
