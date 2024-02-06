import os


def extract_name(file):
    name = os.path.splitext(file)[0].split('/')[8]
    if name.startswith('eval_'):
        return name[:-26]
    return name


# path = '/Users/esteban/Documents/VIQ/VIQ-770/openai-threshold/csvs'

# csv_files = glob.glob(
#    path + '/*.csv')
# file = csv_files[0]

# for file in csv_files:
#     print(extract_name(file))
