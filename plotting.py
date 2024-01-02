import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import ast

# Define categories we want to classify
esg_categories = {
  "emissions": "E",
  "natural resources": "E",
  "pollution": "E",
  "diversity and inclusion": "S",
  "philanthropy": "S",
  "health and safety": "S",
  "training and education": "S",
  "transparancy": "G",
  "corporate compliance": "G",
  "board accountability": "G"}

  
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

def get_dominant_label(labels, scores):
    if not labels or not scores:
        return 'Unknown'
    max_score_index = scores.index(max(scores))
    return labels[max_score_index]

# Apply the function to each row in the DataFrame
aggregated_df['dominant_label'] = aggregated_df.apply(lambda row: get_dominant_label(row['labels'], row['scores']), axis=1)
# Map dominant labels to ESG categories
aggregated_df['dominant_esg'] = aggregated_df['dominant_label'].apply(lambda label: esg_categories.get(label, 'Unknown'))

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

print("Generating bar chart for frequency of dominant labels...")
dominant_label_counts = Counter(aggregated_df['dominant_label'])
plt.figure(figsize=(10, 6))
plt.bar(dominant_label_counts.keys(), dominant_label_counts.values())
plt.title("Frequency of Dominant Labels")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "bar_chart_dominant_labels")

# Count the frequency of each dominant ESG category
dominant_category_counts = Counter(aggregated_df['dominant_esg'])
# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(dominant_category_counts.values(), labels=dominant_category_counts.keys(), autopct='%1.1f%%')
plt.title("Distribution of Dominant ESG Categories")
save_plot_as_pdf(plt, "pie_chart_dominant_esg_distribution")

print("Generating scatter plot for score distributions across dominant labels...")
dominant_scores = aggregated_df['scores'].apply(lambda scores: max(scores) if scores else 0)
plt.figure(figsize=(10, 6))
plt.scatter(aggregated_df['dominant_label'], dominant_scores, alpha=0.5)
plt.title("Score Distributions Across Dominant Labels")
plt.xlabel("Labels")
plt.ylabel("Scores")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "scatter_plot_dominant_score_distribution")


plt.figure(figsize=(12, 8))
sns.boxplot(data=pd.DataFrame([(label, score) for scores, labels in zip(aggregated_df['scores'], aggregated_df['labels']) for score, label in zip(scores, labels)], columns=['Label', 'Score']))
plt.xticks(rotation=45)
plt.title("Box Plot of Score Distribution by Label")
save_plot_as_pdf(plt, "box_plot_score_distribution")

# Count the frequency of each label within each ESG category
label_esg_freq = defaultdict(lambda: defaultdict(int))
for labels, esg in zip(aggregated_df['labels'], aggregated_df['ESG']):
    for label in labels:
        esg_category = esg_categories.get(label, "Unknown")
        label_esg_freq[label][esg_category] += 1

# Convert to DataFrame
label_esg_df = pd.DataFrame(label_esg_freq).fillna(0)

# Plot stacked bar chart
label_esg_df.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title("Label Frequencies by ESG Category")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "stacked_bar_esg_category")

# Flatten the scores and labels for visualization
flat_data = [(label, score) for labels, scores in zip(aggregated_df['labels'], aggregated_df['scores']) for label, score in zip(labels, scores)]
flat_df = pd.DataFrame(flat_data, columns=['Label', 'Score'])

# Create box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Label', y='Score', data=flat_df)
plt.xticks(rotation=45)
plt.title("Box Plot of Score Distribution by Label")
save_plot_as_pdf(plt, "box_plot_score_distribution")

# Count the frequency of each label within each ESG category
label_esg_freq = defaultdict(lambda: defaultdict(int))
for labels, esg in zip(aggregated_df['labels'], aggregated_df['ESG']):
    for label, category in zip(labels, esg):
        label_esg_freq[label][category] += 1

# Convert to DataFrame
label_esg_df = pd.DataFrame(label_esg_freq).fillna(0)

# Plot stacked bar chart
label_esg_df.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title("Label Frequencies by ESG Category")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "stacked_bar_esg_category")


# Count the frequency of each label within each ESG category
label_esg_freq = defaultdict(lambda: defaultdict(int))
for labels, esg in zip(aggregated_df['labels'], aggregated_df['ESG']):
    for label, category in zip(labels, esg):
        label_esg_freq[label][category] += 1

# Convert to DataFrame
label_esg_df = pd.DataFrame(label_esg_freq).fillna(0)

# Plot stacked bar chart
label_esg_df.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title("Label Frequencies by ESG Category")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "stacked_bar_esg_category")

plt.figure(figsize=(12, 8))
sns.violinplot(x='Label', y='Score', data=flat_df)
plt.xticks(rotation=45)
plt.title("Violin Plot of Score Distributions")
save_plot_as_pdf(plt, "violin_plot_score_distributions")

# Prepare data for box plot and violin plot
dominant_label_score_data = pd.DataFrame({'Label': aggregated_df['dominant_label'], 'Score': dominant_scores})

# Box Plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Label', y='Score', data=dominant_label_score_data)
plt.xticks(rotation=45)
plt.title("Box Plot of Score Distribution by Dominant Label")
save_plot_as_pdf(plt, "box_plot_dominant_score_distribution")

# Violin Plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='Label', y='Score', data=dominant_label_score_data)
plt.xticks(rotation=45)
plt.title("Violin Plot of Score Distributions by Dominant Label")
save_plot_as_pdf(plt, "violin_plot_dominant_score_distributions")



plt.close('all')  # Close all open figures
print("All plots generated and saved.")
