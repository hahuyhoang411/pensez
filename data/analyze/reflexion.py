from datasets import load_dataset
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset
data = load_dataset("parquet", data_files="aime25.parquet")

# Function to count occurrences of "wait" in a text (case-insensitive)
def count_wait(text):
    if not isinstance(text, str):
        return 0
    # Count occurrences of "wait" as a whole word (case-insensitive)
    return len(re.findall(r'\bwait\b', text.lower()))

# Process each example in the training set
correct_data = {'prediction_id': [], 'wait_count': []}
incorrect_data = {'prediction_id': [], 'wait_count': []}

for i, example in enumerate(data['train']):
    prediction = example['predictions']
    extractive_match = example['metrics']['extractive_match']
    
    wait_count = count_wait(prediction[0])
    
    if extractive_match == 1:  # Correct prediction
        correct_data['prediction_id'].append(i)
        correct_data['wait_count'].append(wait_count)
    else:  # Incorrect prediction
        incorrect_data['prediction_id'].append(i)
        incorrect_data['wait_count'].append(wait_count)

# Convert to DataFrames for easier analysis
correct_df = pd.DataFrame(correct_data)
incorrect_df = pd.DataFrame(incorrect_data)

# Calculate statistics
total_correct = len(correct_df)
total_incorrect = len(incorrect_df)
total_wait_correct = correct_df['wait_count'].sum()
total_wait_incorrect = incorrect_df['wait_count'].sum()
avg_wait_correct = total_wait_correct / total_correct if total_correct > 0 else 0
avg_wait_incorrect = total_wait_incorrect / total_incorrect if total_incorrect > 0 else 0

# Create a summary DataFrame
summary = pd.DataFrame({
    'Category': ['Correct (extractive_match=1)', 'Incorrect (extractive_match=0)'],
    'Count': [total_correct, total_incorrect],
    'Total Wait Words': [total_wait_correct, total_wait_incorrect],
    'Average Wait Per Prediction': [avg_wait_correct, avg_wait_incorrect]
})

# Create distribution DataFrame for plotting
correct_df['category'] = 'Correct'
incorrect_df['category'] = 'Incorrect'
plot_df = pd.concat([correct_df, incorrect_df])

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Bar chart comparing averages
plt.subplot(2, 2, 1)
plt.bar(['Correct', 'Incorrect'], [avg_wait_correct, avg_wait_incorrect], color=['green', 'red'])
plt.title('Average "Wait" Occurrences per Prediction')
plt.ylabel('Average Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 2. Box plot showing distribution
plt.subplot(2, 2, 2)
sns.boxplot(x='category', y='wait_count', data=plot_df, palette=['green', 'red'])
plt.title('Distribution of "Wait" Occurrences')
plt.xlabel('')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 3. Histogram of wait counts by category
plt.subplot(2, 2, 3)
bins = np.arange(0, max(plot_df['wait_count']) + 2) - 0.5
sns.histplot(data=plot_df, x='wait_count', hue='category', multiple='dodge', 
             bins=bins, palette=['green', 'red'], discrete=True)
plt.title('Histogram of "Wait" Occurrences')
plt.xlabel('Number of "Wait" in Prediction')
plt.ylabel('Count of Predictions')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 4. Scatter plot of prediction ID vs wait count
plt.subplot(2, 2, 4)
plt.scatter(correct_df['prediction_id'], correct_df['wait_count'], 
            label='Correct', color='green', alpha=0.7)
plt.scatter(incorrect_df['prediction_id'], incorrect_df['wait_count'], 
            label='Incorrect', color='red', alpha=0.7)
plt.title('Prediction ID vs "Wait" Count')
plt.xlabel('Prediction ID')
plt.ylabel('Wait Count')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('wait_analysis_comparison.png')
plt.show()

# Print numerical summary
print("Summary Statistics:")
print(summary)

# Calculate correlation between wait count and correctness
correct_vals = correct_df['wait_count'].tolist()
incorrect_vals = incorrect_df['wait_count'].tolist()

print("\nStatistical Analysis:")
print(f"Correct predictions: mean={np.mean(correct_vals):.2f}, median={np.median(correct_vals):.2f}, max={np.max(correct_vals)}")
print(f"Incorrect predictions: mean={np.mean(incorrect_vals):.2f}, median={np.median(incorrect_vals):.2f}, max={np.max(incorrect_vals)}")

# t-test to check if the difference is statistically significant
from scipy import stats
t_stat, p_val = stats.ttest_ind(correct_vals, incorrect_vals, equal_var=False)
print(f"\nStatistical significance (t-test): t={t_stat:.2f}, p={p_val:.4f}")
print(f"Interpretation: {'Statistically significant difference' if p_val < 0.05 else 'No statistically significant difference'}")

# Count how many predictions have at least one "wait"
correct_with_wait = sum(1 for count in correct_vals if count > 0)
incorrect_with_wait = sum(1 for count in incorrect_vals if count > 0)
print(f"\nPredictions with at least one 'wait':")
print(f"Correct: {correct_with_wait}/{total_correct} ({correct_with_wait/total_correct*100:.1f}%)")
print(f"Incorrect: {incorrect_with_wait}/{total_incorrect} ({incorrect_with_wait/total_incorrect*100:.1f}%)")