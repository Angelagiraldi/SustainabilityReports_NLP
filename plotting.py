import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
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
    "board accountability": "G"
}

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

# Function to create and save box and violin plots
def create_distribution_plots(data, title_prefix, filename_prefix):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Label', y='Score', data=data)
    plt.xticks(rotation=45)
    plt.title(f"{title_prefix} Box Plot of Score Distribution")
    save_plot_as_pdf(plt, f"{filename_prefix}_box_plot_score_distribution")

    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Label', y='Score', data=data)
    plt.xticks(rotation=45)
    plt.title(f"{title_prefix} Violin Plot of Score Distributions")
    save_plot_as_pdf(plt, f"{filename_prefix}_violin_plot_score_distributions")

# Generating heatmap of label scores for 20 random sentences
print("Generating heatmap of label scores for 20 random sentences...")
sampled_df = aggregated_df.sample(n=20)

# Create a list of dictionaries, each containing label-score pairs for a sentence
heatmap_data_list = []
for _, row in sampled_df.iterrows():
    label_score_dict = {label: score for label, score in zip(row['labels'], row['scores'])}
    heatmap_data_list.append(label_score_dict)

# Convert the list of dictionaries to a DataFrame
heatmap_data = pd.DataFrame(heatmap_data_list, index=sampled_df['sentence'])
print(heatmap_data.head())  # To see the first few rows of the DataFrame

# Replace NaN with 0 for better visualization
heatmap_data.fillna(0, inplace=True)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
plt.title("Heatmap of Label Scores for 20 Random Sentences")
plt.xticks(rotation=45)
plt.tight_layout()
save_plot_as_pdf(plt, "heatmap_label_scores")

# Generating bar chart for frequency of dominant labels
print("Generating bar chart for frequency of dominant labels...")
dominant_label_counts = Counter(aggregated_df['dominant_label'])
plt.figure(figsize=(10, 6))
plt.bar(dominant_label_counts.keys(), dominant_label_counts.values())
plt.title("Frequency of Dominant Labels")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "bar_chart_dominant_labels")

# Generating pie chart for distribution of dominant ESG categories
print("Generating pie chart for distribution of dominant ESG categories...")
dominant_category_counts = Counter(aggregated_df['dominant_esg'])
plt.figure(figsize=(8, 8))
plt.pie(dominant_category_counts.values(), labels=dominant_category_counts.keys(), autopct='%1.1f%%')
plt.title("Distribution of Dominant ESG Categories")
save_plot_as_pdf(plt, "pie_chart_dominant_esg_distribution")

# Generating scatter plot for score distributions across dominant labels
print("Generating scatter plot for score distributions across dominant labels...")
dominant_scores = aggregated_df['scores'].apply(lambda scores: max(scores) if scores else 0)
plt.figure(figsize=(10, 6))
plt.scatter(aggregated_df['dominant_label'], dominant_scores, alpha=0.5)
plt.title("Score Distributions Across Dominant Labels")
plt.xlabel("Labels")
plt.ylabel("Scores")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "scatter_plot_dominant_score_distribution")

# Create box and violin plots for all labels
flat_df = pd.DataFrame([(label, score) for labels, scores in zip(aggregated_df['scores'], aggregated_df['labels']) for label, score in zip(labels, scores)], columns=['Label', 'Score'])
create_distribution_plots(flat_df, "All Labels", "all_labels")

# Create box and violin plots for dominant labels
dominant_label_score_data = pd.DataFrame({'Label': aggregated_df['dominant_label'], 'Score': dominant_scores})
create_distribution_plots(dominant_label_score_data, "Dominant Label", "dominant_label")


# Initialize a dictionary to store the frequencies
dominant_label_esg_freq = defaultdict(lambda: defaultdict(int))

# Iterate over the DataFrame to populate the dictionary
for _, row in aggregated_df.iterrows():
    dominant_label_esg_freq[row['dominant_esg']][row['dominant_label']] += 1

# Convert the dictionary to a DataFrame
dominant_label_esg_df = pd.DataFrame(dominant_label_esg_freq).fillna(0)
# Plot stacked bar chart
plt.figure(figsize=(12, 8))
dominant_label_esg_df.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title("Frequency of Dominant Labels within ESG Categories")
plt.xlabel("ESG Categories")
plt.ylabel("Frequency of Dominant Labels")
plt.xticks(rotation=45)
save_plot_as_pdf(plt, "stacked_bar_dominant_label_esg")

plt.close('all')
print("All plots generated and saved.")
