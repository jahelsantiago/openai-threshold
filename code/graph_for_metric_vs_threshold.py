from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from extract_name import extract_name


def plot_metrics(file):
    # Load the data
    data = pd.read_csv(file)
    name = extract_name(file)

    # Split the data
    X = data['cosine_distance']
    y = []

    for i in range(len(X)):
        if type(data['gpt_evaluation'][i]) is str:
            y.append(1) if data['gpt_evaluation'][i].contains('True') else y.append(0)
        else:
            y.append(1) if data['gpt_evaluation'][i] else y.append(0)

    # Calculate Cosine Similarity
    X = 1 - X

    # Initialize lists to store metrics
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    cross_entropies = []
    thresholds = np.linspace(0.7, 0.85, 100)

    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Make predictions based on the threshold
        y_pred = (X > threshold).astype(int)

        # Calculate metrics
        precisions.append(precision_score(y, y_pred))
        recalls.append(recall_score(y, y_pred))
        f1_scores.append(f1_score(y, y_pred))
        roc_aucs.append(roc_auc_score(y, y_pred))
        cross_entropies.append(log_loss(y, y_pred))

    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, roc_aucs, label='ROC AUC')
    # plt.plot(thresholds, cross_entropies, label='Cross Entropy')
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics for Different Thresholds:' + name)

    # plt.show()
    # Save the figure
    plt.savefig('/Users/esteban/Documents/VIQ/VIQ-770/openai-threshold/Plots/'
                + name + '_metrics.png')

    plt.close()

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # plt.show()
    # Save the figure
    plt.savefig(name + '_roc.png')

    plt.close()
