import csv

with open('eval_10K_software_prompt_7_gpt_3_5_balanced.csv') as file_object:
    distribution_dict = {
        '>0.0 - <0.06': {
            'count': 0
        },
        '>0.06 - <0.1': {
            'count': 0
        },
        '>0.1 - <0.16': {
            'count': 0
        },
        '>0.16 - <0.2': {
            'count': 0
        },
        '>0.2 - <0.22': {
            'count': 0
        },
        '>0.22 - <0.24': {
            'count': 0
        },
        '>0.24 - <0.26': {
            'count': 0
        },
        '>0.26 - <0.28': {
            'count': 0
        },
        '>0.28 - <0.3': {
            'count': 0
        },
        '>0.3 - <0.36': {
            'count': 0
        },
        '>0.36 - <0.4': {
            'count': 0
        }
    }
    reader_obj = csv.DictReader(file_object)

    for row in reader_obj:
        cosine = float(row['cosine_distance'])

        if cosine >= 0 and cosine < 0.06:
            distribution_dict['>0.0 - <0.06']['count'] += 1
        elif cosine >= 0.06 and cosine < 0.1:
            distribution_dict['>0.06 - <0.1']['count'] += 1
        elif cosine >= 0.1 and cosine < 0.16:
            distribution_dict['>0.1 - <0.16']['count'] += 1
        elif cosine >= 0.16 and cosine < 0.2:
            distribution_dict['>0.16 - <0.2']['count'] += 1
        elif cosine >= 0.2 and cosine < 0.22:
            distribution_dict['>0.2 - <0.22']['count'] += 1
        elif cosine >= 0.22 and cosine < 0.24:
            distribution_dict['>0.22 - <0.24']['count'] += 1
        elif cosine >= 0.24 and cosine < 0.26:
            distribution_dict['>0.24 - <0.26']['count'] += 1
        elif cosine >= 0.26 and cosine < 0.28:
            distribution_dict['>0.26 - <0.28']['count'] += 1
        elif cosine >= 0.28 and cosine < 0.3:
            distribution_dict['>0.28 - <0.3']['count'] += 1
        elif cosine >= 0.3 and cosine < 0.36:
            distribution_dict['>0.3 - <0.36']['count'] += 1
        elif cosine >= 0.36 and cosine < 0.4:
            distribution_dict['>0.36 - <0.4']['count'] += 1

    print(distribution_dict)


import matplotlib.pyplot as plt
import numpy as np

# Sample data
ranges = [
    '.0-.06',
    '.06-.1',
    '.1-.16',
    '.16-.2',
    '.2-.22',
    '.22-.24',
    '.24-.26',
    '.26-.28',
    '.28-.3',
    '.3-.36',
    '.36-.4'
]
true_values = [distribution_dict[val]['count'] for val in distribution_dict]

# Calculate the x-axis positions for the bars
x = np.arange(len(ranges))
plt.figure(figsize=(10, 6))
# Plotting the bars
plt.bar(x, true_values, width=0.4, label='count')

# Adding labels and title
plt.xlabel('Ranges')
plt.ylabel('Count')
plt.title('results count by range')
plt.xticks(x, ranges)

# Adding legend
plt.legend()

# Show the plot
plt.savefig("scores_dist_software_10K_balanced_lower_ranges.png")
