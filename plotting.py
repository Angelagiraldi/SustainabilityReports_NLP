import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

file_path = "aggregated_results.csv"  # Path to your CSV file
aggregated_df = pd.read_csv(file_path)

# Since 'labels', 'scores', and 'ESG' are likely saved as strings, you might need to convert them back to lists
import ast
aggregated_df['labels'] = aggregated_df['labels'].apply(ast.literal_eval)
aggregated_df['scores'] = aggregated_df['scores'].apply(ast.literal_eval)
aggregated_df['ESG'] = aggregated_df['ESG'].apply(ast.literal_eval)

# Sample 20 sentences
sampled_df = aggregated_df.sample(n=20)

# Create a DataFrame suitable for a heatmap
heatmap_data = pd.DataFrame(sampled_df['scores'].tolist(), index=sampled_df['sentence'], columns=sampled_df['labels'].iloc[0])

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
plt.title("Heatmap of Label Scores for 20 Random Sentences")
plt.show()

# Count the frequency of each label
label_counts = Counter(label for labels in aggregated_df['labels'] for label in labels)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(label_counts.keys(), label_counts.values())
plt.title("Frequency of Top Labels")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Count the frequency of each ESG category
category_counts = Counter(category for categories in aggregated_df['ESG'] for category in categories)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
plt.title("Distribution of ESG Categories")
plt.show()


# Flatten the scores and corresponding labels
flat_scores = [score for scores in aggregated_df['scores'] for score in scores]
flat_labels = [label for labels in aggregated_df['labels'] for label in labels]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(flat_labels, flat_scores, alpha=0.5)
plt.title("Score Distributions Across Labels")
plt.xlabel("Labels")
plt.ylabel("Scores")
plt.xticks(rotation=45)
plt.show()