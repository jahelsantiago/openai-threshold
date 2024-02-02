import glob
from graph_for_metric_vs_threshold import plot_metrics
from histogram_companies_vs_cosine_similarity import plot_histogram
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

# file = 'freight visibility software-1k.csv'
# plot_histogram(file)
