import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_histogram(file):
    # Load the data
    data = pd.read_csv(file)

    # Split the data
    X = data['cosine_distance']

    X = 1 - X

    # Define bin edges
    bin_edges = np.linspace(0.70, 0.85, 21)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(X, bins=bin_edges, alpha=0.5, color='g')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarity'
              + ' ' + os.path.splitext(file)[0])

    # Save the figure
    plt.savefig(os.path.splitext(file)[0] + '_histogram.png')

    plt.close()
