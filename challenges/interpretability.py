import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define prediction function
def predict_probabilities(texts):
    # Debugging: Check the input type and content
    print(f"Input type: {type(texts)}")
    print(f"Input content: {texts}")

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

# Define sample text
sample_text = ["Women belong to kitchen", "Today is a sunny day"]
print(f"Sample text: {sample_text}")

# Generate SHAP values
shap_values = explainer(sample_text)
print(f"SHAP values: {shap_values}")

# Visualize SHAP values using a waterfall plot for the first sample, class 1 ("sexist")
shap.waterfall_plot(shap.Explanation(
    values=shap_values.values[0][:, 1],         # Contributions for class 1
    base_values=shap_values.base_values[0][1], # Base value for class 1
    data=shap_values.data[0]                   # Tokens for the first input
))

# Visualize SHAP values for the second sample, class 1 ("sexist")
shap.waterfall_plot(shap.Explanation(
    values=shap_values.values[1][:, 1],         # Contributions for class 1
    base_values=shap_values.base_values[1][1], # Base value for class 1
    data=shap_values.data[1]                   # Tokens for the second input
))

# Save the SHAP plots
plt.savefig("shap_visualization_sample1.png", dpi=300, bbox_inches="tight")
plt.savefig("shap_visualization_sample2.png", dpi=300, bbox_inches="tight")
plt.show()
