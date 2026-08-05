#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import itertools

# Load data
df = pd.read_excel("testing 4 features.xlsx")

# Define triangular membership function
def safe_triangular_mf(x, a, b, c):
    if a == b:
        return 1.0 if a <= x <= c else 0.0
    elif b == c:
        return 1.0 if a <= x <= c else 0.0
    elif x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    else:
        return 0.0

# Step 1: Calculate quantiles for membership function ranges
features = ['Channel 2 RMS', 'Channel 2 PEAK', 'Channel 2 KURTOSIS', 'Channel 2 PSD']
q_stats = {}
for feature in features:
    q_stats[feature] = {
        'min': df[feature].min(),
        'q1': df[feature].quantile(0.25),
        'q2': df[feature].quantile(0.50),
        'q3': df[feature].quantile(0.75),
        'mean': df[feature].mean(),
        'max': df[feature].max()
    }

# Step 2: Define membership functions for each feature (Low, Medium, High)
feature_mfs_3lvl = {
    'Channel 2 RMS': {
        'Low':   (q_stats['Channel 2 RMS']['min'], q_stats['Channel 2 RMS']['min'], q_stats['Channel 2 RMS']['mean']),
        'Medium':(q_stats['Channel 2 RMS']['q1'], q_stats['Channel 2 RMS']['q2'], q_stats['Channel 2 RMS']['q3']),
        'High':  (q_stats['Channel 2 RMS']['mean'], q_stats['Channel 2 RMS']['max'], q_stats['Channel 2 RMS']['max'])
    },
    'Channel 2 PEAK': {
        'Low':   (q_stats['Channel 2 PEAK']['min'], q_stats['Channel 2 PEAK']['min'], q_stats['Channel 2 PEAK']['mean']),
        'Medium':(q_stats['Channel 2 PEAK']['q1'], q_stats['Channel 2 PEAK']['q2'], q_stats['Channel 2 PEAK']['q3']),
        'High':  (q_stats['Channel 2 PEAK']['mean'], q_stats['Channel 2 PEAK']['max'], q_stats['Channel 2 PEAK']['max'])
    },
    'Channel 2 KURTOSIS': {
        'Low':   (q_stats['Channel 2 KURTOSIS']['min'], q_stats['Channel 2 KURTOSIS']['min'], q_stats['Channel 2 KURTOSIS']['mean']),
        'Medium':(q_stats['Channel 2 KURTOSIS']['q1'], q_stats['Channel 2 KURTOSIS']['q2'], q_stats['Channel 2 KURTOSIS']['q3']),
        'High':  (q_stats['Channel 2 KURTOSIS']['mean'], q_stats['Channel 2 KURTOSIS']['max'], q_stats['Channel 2 KURTOSIS']['max'])
    },
    'Channel 2 PSD': {
        'Low':   (q_stats['Channel 2 PSD']['min'], q_stats['Channel 2 PSD']['min'], q_stats['Channel 2 PSD']['mean']),
        'Medium':(q_stats['Channel 2 PSD']['q1'], q_stats['Channel 2 PSD']['q2'], q_stats['Channel 2 PSD']['q3']),
        'High':  (q_stats['Channel 2 PSD']['mean'], q_stats['Channel 2 PSD']['max'], q_stats['Channel 2 PSD']['max'])
    }
}

# Step 3: Output values
output_map = {'Smooth': 1.9970, 'Average': 2.3511, 'Rough': 2.7480}
levels_3 = ['Low', 'Medium', 'High']

# Step 4: Generate 81 fuzzy rules
rule_combinations_3lvl = list(itertools.product(levels_3, repeat=4))

def determine_output_label(combo):
    counts = {'Low': 0, 'Medium': 0, 'High': 0}
    for level in combo:
        counts[level] += 1
    if counts['High'] >= 3:
        return 'Rough'
    elif counts['Low'] >= 3:
        return 'Smooth'
    else:
        return 'Average'

rule_base_81 = []
for combo in rule_combinations_3lvl:
    output_label = determine_output_label(combo)
    rule_base_81.append({
        'Inputs': combo,
        'Output Label': output_label,
        'Output Value': output_map[output_label]
    })

# Step 5: Apply fuzzy logic for all parts
predictions_81 = []
accuracies_81 = []

for _, row in df.iterrows():
    # Calculate degrees
    membership_degrees = {}
    for feature in features:
        val = row[feature]
        membership_degrees[feature] = {
            level: safe_triangular_mf(val, *feature_mfs_3lvl[feature][level]) for level in levels_3
        }

    # Evaluate rules (min composition)
    rule_outputs = []
    for rule in rule_base_81:
        combo = rule['Inputs']
        strength = min([
            membership_degrees[features[i]][combo[i]] for i in range(4)
        ])
        rule_outputs.append((strength, rule['Output Value']))

    # Defuzzify
    num = sum(fs * val for fs, val in rule_outputs)
    den = sum(fs for fs, _ in rule_outputs)
    y_pred = num / den if den > 0 else output_map['Average']

    # Accuracy
    y_true = row['Y_true']
    accuracy = 100 - (abs(y_true - y_pred) / y_true) * 100

    predictions_81.append(round(y_pred, 4))
    accuracies_81.append(round(accuracy, 2))

# Final output
df['Y_pred'] = predictions_81
df['Accuracy (%)'] = accuracies_81

# Save results to Excel (optional)
df.to_excel("Fuzzy_Rule_Output.xlsx", index=False)

# Preview
print(df[['Y_true', 'Y_pred', 'Accuracy (%)']])


# In[ ]:




