import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from extract_name import extract_name


def plot_histogram(file):
    # Load the data
    data = pd.read_csv(file)
    name = extract_name(file)

    # Split the data
    X = data['cosine_distance']

    X = 1 - X

    # Define bin edges
    bin_edges = np.arange(0, 1, 0.01)

    # Plot histogram
    plt.figure(figsize=(40, 6))
    results = plt.hist(X, bins=bin_edges, alpha=0.5, color='g', density=True)

    mid_points = (bin_edges[:-1] + bin_edges[1:]) / 2

    expected_value = (mid_points * results[0]).sum()/results[0].sum()

    bin_index = np.digitize(expected_value, bin_edges) - 1

    expected_value_bin = bin_edges[bin_index]

    second_moment = (mid_points**2 * results[0]).sum()/results[0].sum()

    variance = second_moment - expected_value_bin**2

    plt.xticks(bin_edges)
    plt.grid()
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarity: ' + name)

    # Save the figure
    plt.savefig('/Users/esteban/Documents/VIQ/VIQ-770/openai-threshold/plots/'
                + name + '_histogram.png')

    plt.close()

    print(f'The mean is at cosine_similarity = {expected_value_bin}')
    print('The standard deviation is = ', np.sqrt(variance))
