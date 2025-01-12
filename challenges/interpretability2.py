import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_name = "NLP-LTU/bertweet-large-sexism-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define prediction function
def predict_probabilities(texts):
    # Convert SHAP input (e.g., numpy.ndarray) to a list of strings
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    # Ensure `texts` is a list of strings
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("Input must be a string or a list of strings.")

    # Tokenize and generate predictions
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**tokens)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.detach().numpy()

# Initialize SHAP explainer
explainer = shap.Explainer(predict_probabilities, tokenizer)

# Example corpus (replace with your dataset)
corpus = [
    "Women belong to kitchen",
    "Today is a sunny day",
    "She should not work",
    "Equality is essential",
    "Men can also cook",
]

# Generate SHAP values for the corpus
shap_values_corpus = explainer(corpus)

# Aggregate SHAP values across tokens
token_contributions = defaultdict(float)
token_frequencies = defaultdict(int)

for i, tokens in enumerate(shap_values_corpus.data):  # Tokens for each sentence
    for j, token in enumerate(tokens):  # Iterate over tokens in the sentence
        token_contributions[token] += shap_values_corpus.values[i][j, 1]  # Class 1 SHAP value
        token_frequencies[token] += 1

# Compute mean contribution for each token
mean_contributions = {
    token: contrib / token_frequencies[token] for token, contrib in token_contributions.items()
}

# Visualize top tokens contributing to "sexist" predictions
sorted_tokens = sorted(mean_contributions.items(), key=lambda x: x[1], reverse=True)
top_tokens = sorted_tokens[:10]  # Top 10 tokens
tokens, contributions = zip(*top_tokens)

plt.figure(figsize=(10, 6))
plt.barh(tokens, contributions, color='steelblue')
plt.xlabel("Mean SHAP Contribution")
plt.ylabel("Tokens")
plt.title("Top Tokens Contributing to 'Sexist' Predictions")
plt.gca().invert_yaxis()
plt.show()

# Visualize SHAP values for specific sentences
for i, text in enumerate(corpus):
    print(f"Sentence: {text}")
    shap.waterfall_plot(shap.Explanation(
        values=shap_values_corpus.values[i][:, 1],         # Contributions for class 1 ("sexist")
        base_values=shap_values_corpus.base_values[i][1], # Base value for class 1
        data=shap_values_corpus.data[i]                   # Tokens for the input
    ))

# Save visualizations for the first sentence
plt.savefig("shap_visualization_sentence1.png", dpi=300, bbox_inches="tight")

# Analyze sensitive tokens
sensitive_tokens = ["Women", "Men", "kitchen", "cook"]
for token in sensitive_tokens:
    if token in mean_contributions:
        print(f"Token: {token}")
        print(f"  Mean Contribution: {mean_contributions[token]:.4f}")
        print(f"  Frequency: {token_frequencies[token]}")

# Example: True labels for the corpus (replace with actual labels)
true_labels = ["sexist", "not sexist", "sexist", "not sexist", "not sexist"]

# Identify misclassified examples
probs = predict_probabilities(corpus)  # Predicted probabilities
predicted_labels = ["sexist" if p[1] > 0.5 else "not sexist" for p in probs]

for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
    if true != pred:
        print(f"Misclassified Sentence: {corpus[i]}")
        print(f"  True Label: {true}")
        print(f"  Predicted Label: {pred}")
        # Visualize SHAP values for this misclassified sentence
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_corpus.values[i][:, 1],
            base_values=shap_values_corpus.base_values[i][1],
            data=shap_values_corpus.data[i]
        ))
