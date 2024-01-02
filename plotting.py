import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import ast

print("Reading and preprocessing the DataFrame...")
file_path = "aggregated_results.csv"
aggregated_df = pd.read_csv(file_path)
aggregated_df['labels'] = aggregated_df['labels'].apply(ast.literal_eval)
aggregated_df['scores'] = aggregated_df['scores'].apply(ast.literal_eval)
aggregated_df['ESG'] = aggregated_df['ESG'].apply(ast.literal_eval)

# Function to save plot as PDF
def save_plot_as_pdf(plt, filename):
    plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')
    print(f"Saved plot as {filename}.pdf")

print("Generating heatmap of label scores for 20 random sentences...")

# Get all unique labels across the dataset
all_labels = set(label for labels in aggregated_df['labels'] for label in labels)

# Sample 20 sentences
sampled_df = aggregated_df.sample(n=20)

# Create a DataFrame suitable for a heatmap
heatmap_data = pd.DataFrame(index=sampled_df['sentence'])

for label in all_labels:
    # For each label, create a column in the DataFrame
    heatmap_data[label] = sampled_df.apply(lambda row: row['scores'][row['labels'].index(label)] if label in row['labels'] else float('nan'), axis=1)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
plt.title("Heatmap of Label Scores for 20 Random Sentences")
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to fit the x-axis labels
save_plot_as_pdf(plt, "heatmap_label_scores")


# Bar Chart for Frequency of Top Labels
print("Generating bar chart for frequency of top labels...")
label_counts = Counter(label for labels in aggregated_df['labels'] for label in labels)
plt.figure(figsize=(10, 6))
plt.bar(label_counts.keys(), label_counts.values())
plt.title("Frequency of Top Labels")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "bar_chart_top_labels")

# Pie Chart for Distribution of ESG Categories
print("Generating pie chart for distribution of ESG categories...")
category_counts = Counter(category for categories in aggregated_df['ESG'] for category in categories)
plt.figure(figsize=(8, 8))
plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
plt.title("Distribution of ESG Categories")
save_plot_as_pdf(plt, "pie_chart_esg_distribution")

# Scatter Plot for Score Distributions Across Labels
print("Generating scatter plot for score distributions across labels...")
flat_scores = [score for scores in aggregated_df['scores'] for score in scores]
flat_labels = [label for labels in aggregated_df['labels'] for label in labels]
plt.figure(figsize=(10, 6))
plt.scatter(flat_labels, flat_scores, alpha=0.5)
plt.title("Score Distributions Across Labels")
plt.xlabel("Labels")
plt.ylabel("Scores")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "scatter_plot_score_distribution")


plt.figure(figsize=(12, 8))
sns.boxplot(data=pd.DataFrame([(label, score) for scores, labels in zip(aggregated_df['scores'], aggregated_df['labels']) for score, label in zip(scores, labels)], columns=['Label', 'Score']))
plt.xticks(rotation=45)
plt.title("Box Plot of Score Distribution by Label")
save_plot_as_pdf(plt, "box_plot_score_distribution")

label_esg_counts = pd.DataFrame([(label, esg_categories[label], 1) for labels in aggregated_df['labels'] for label in labels], columns=['Label', 'ESG', 'Count']).groupby(['Label', 'ESG']).count().unstack().fillna(0)
label_esg_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title("Label Frequencies by ESG Category")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "stacked_bar_esg_category")

sns.heatmap(label_features.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Labels")
save_plot_as_pdf(plt, "correlation_heatmap_labels")


plt.close('all')  # Close all open figures
print("All plots generated and saved.")
