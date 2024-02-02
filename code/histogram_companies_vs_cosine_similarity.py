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
    bin_edges = np.linspace(0.5, 0.85, 21)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(X, bins=bin_edges, alpha=0.5, color='g')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarity: ' + name)

    # Save the figure
    plt.savefig('/Users/esteban/Documents/VIQ/VIQ-770/openai-threshold/plots/'
                + name + '_histogram.png')

    plt.close()
