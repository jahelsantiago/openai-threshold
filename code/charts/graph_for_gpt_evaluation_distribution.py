import csv

with open('eval_10K_software_prompt_7_gpt_3_5_balanced.csv') as file_object:
    distribution_dict = {
        '>0.0 - <0.06': {
            'true': 0,
            'false': 0
        },
        '>0.06 - <0.1': {
            'true': 0,
            'false': 0
        },
        '>0.1 - <0.16': {
            'true': 0,
            'false': 0
        },
        '>0.16 - <0.2': {
            'true': 0,
            'false': 0
        },
        '>0.2 - <0.22': {
            'true': 0,
            'false': 0
        },
        '>0.22 - <0.24': {
            'true': 0,
            'false': 0
        },
        '>0.24 - <0.26': {
            'true': 0,
            'false': 0
        },
        '>0.26 - <0.28': {
            'true': 0,
            'false': 0
        },
        '>0.28 - <0.3': {
            'true': 0,
            'false': 0
        },
        '>0.3 - <0.36': {
            'true': 0,
            'false': 0
        },
        '>0.36 - <0.4': {
            'true': 0,
            'false': 0
        }
    }
    
    reader_obj = csv.DictReader(file_object)

    for row in reader_obj:
        cosine = float(row['cosine_distance'])
        gpt_eval = row['gpt_evaluation']

        if cosine >= 0 and cosine < 0.06:
            if gpt_eval == 'True':
                distribution_dict['>0.0 - <0.06']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.0 - <0.06']['false'] += 1
        elif cosine >= 0.06 and cosine < 0.1:
            if gpt_eval == 'True':
                distribution_dict['>0.06 - <0.1']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.06 - <0.1']['false'] += 1
        elif cosine >= 0.1 and cosine < 0.16:
            if gpt_eval == 'True':
                distribution_dict['>0.1 - <0.16']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.1 - <0.16']['false'] += 1
        elif cosine >= 0.16 and cosine < 0.2:
            if gpt_eval == 'True':
                distribution_dict['>0.16 - <0.2']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.16 - <0.2']['false'] += 1
        elif cosine >= 0.2 and cosine < 0.22:
            if gpt_eval == 'True':
                distribution_dict['>0.2 - <0.22']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.2 - <0.22']['false'] += 1
        elif cosine >= 0.22 and cosine < 0.24:
            if gpt_eval == 'True':
                distribution_dict['>0.22 - <0.24']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.22 - <0.24']['false'] += 1
        elif cosine >= 0.24 and cosine < 0.26:
            if gpt_eval == 'True':
                distribution_dict['>0.24 - <0.26']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.24 - <0.26']['false'] += 1
        elif cosine >= 0.26 and cosine < 0.28:
            if gpt_eval == 'True':
                distribution_dict['>0.26 - <0.28']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.26 - <0.28']['false'] += 1
        elif cosine >= 0.28 and cosine < 0.3:
            if gpt_eval == 'True':
                distribution_dict['>0.28 - <0.3']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.28 - <0.3']['false'] += 1
        elif cosine >= 0.3 and cosine < 0.36:
            if gpt_eval == 'True':
                distribution_dict['>0.3 - <0.36']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.3 - <0.36']['false'] += 1
        elif cosine >= 0.36 and cosine < 0.4:
            if gpt_eval == 'True':
                distribution_dict['>0.36 - <0.4']['true'] += 1
            elif gpt_eval == 'False':
                distribution_dict['>0.36 - <0.4']['false'] += 1

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
true_values = [distribution_dict[val]['true'] for val in distribution_dict]
false_values = [distribution_dict[val]['false'] for val in distribution_dict]

# Calculate the x-axis positions for the bars
x = np.arange(len(ranges))
plt.figure(figsize=(10, 6))
# Plotting the bars
plt.bar(x - 0.2, true_values, width=0.4, label='True')
plt.bar(x + 0.2, false_values, width=0.4, label='False')

# Adding labels and title
plt.xlabel('Ranges')
plt.ylabel('Count')
plt.title('True vs False Results by Range')
plt.xticks(x, ranges)

# Adding legend
plt.legend()

# Show the plot
plt.savefig("data_dist_software_final_10K_balanced_lower_ranges.png")
