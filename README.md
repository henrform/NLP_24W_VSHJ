# NLP_24W_VSHJ
Topic: Detection of Online Sexism

Students: Sofija Milicic, Henry Werner Forman, Vanja Draganic, Johannes Pfennigbauer

We are using Python 3.10. The necessary packages can be found in `requirements.txt`.

#### Milestone 1
Note: The `data` folder contains 2 datasets: one with individual labels from 3 annotators per comment and another aggregated version, where 2/3 agreements were resolved by the dataset creators. For preprocessing, we used only the aggregated dataset to avoid redundant processing. The presence of 2/3 agreements will be revisited during the error analysis in milestone 2.

A detailed exploratory analysis, including text normalization steps and explanations for the preprocessing decisions, is documented in `milestone 1/milestone_1.ipynb`. The entire preprocessing workflow, including conclusions from the analysis and conversion to the standard CoNLL-U format, is implemented in the `TextProcessingPipeline` class, located in the script `milestone 1/pipeline_connl.py`. The dataset, exported in the standard format and split into training, test, and validation sets, is available in the `data/processed` folder.

Outlined below is a diagram of the preprocessing steps implemented in the `TextProcessingPipeline` class.
<img src="assets/preprocessing_pipeline.png" width="80%"/>

#### Milestone 2

All relevant code is located in the `milestone 2` folder, which includes:

1. `milestone_2.ipynb` - a notebook inside which various baseline models are trained and tested, results are summarized and qualitative analysis is conducted
2. `import_preprocess.py` - with `ImportPreprocess` class whose methods handle reading CoNLL-U files, balancing the dataset and extracting features (BoW and TF-IDF)
3. `baseline_models.py` - defines separate classes for various baseline models, all of which inherit evaluation and qualitative analysis methods from the abstract base class `ClassificationModel`
4. `preprocess_input.py` - provides a function for preprocessing input sentences in the same way as in `milestone 1/pipeline_connl.py`, but without generating CoNLL-U files. This was implemented with the aim of testing the models with the sentences we come up with, to examine how they're classified

Baseline models that we implemented:

a) traditional (non-DL)

1. Majority Class - ` MajorityClassClassifier` class
2. Rule-based (regex) - `RuleBasedClassifier` class
3. Naive Bayes - `NaiveBayesClassifier` class
4. Logistic Regression - `LogisticRegression` class
5. XGBoost (using BoW/TF-IDF features) - `XGBoostClassifier` class

b) DL baseline models

1. LSTM - `LSTM_Model` class


The dataset is imbalanced, so in `milestone_2.ipynb`, all models are trained using both the original dataset and the balanced version. Metrics: accuracy, balanced accuracy, precision and recall are tracked and presented alongside confusion matrices in each case. The `Results` section of the notebook provides a summary of the outcomes for both scenarios: training with balanced and unbalanced data for each model tested.

`Qualitative/error analysis` was performed to thoroughly assess the quality and limitations of the baseline models' predictions, as well as to inspire us with ideas for improvement. The analysis primarily focuses on the LSTM and Naive Bayes models. Functionalities that we implemented:

1. randomly sampling sentences from each of the 4 corners of confusion matrix (TN, FP, FN, TP) and examining them alongside individual labels from 3 annotators
2. extracting the most frequently occurring tokens for each of the 4 corners of the confusion matrix
3. identifying the most common tokens in the context of a specified fixed token
4. testing the models on custom sentences

Baseline models offer a solid starting point but have notable limitations. They struggle to capture context, deeper semantic meaning, sarcasm or implicit sexism. Despite balancing efforts, they remain biased toward the majority class and show limited generalizability, relying heavily on explicit patterns and word frequencies.

#### Update after the review meeting
As discussed, the problem of sexism detection is ambiguous and it's worth experimenting with different techniques for aggregating the individual annotations from the 3 annotators. This type of analysis can be found in the `milestone 2/milestone_2.ipynb` under the section `Different methods of aggregation`. Beyond the original labels provided by the dataset creators, we explored the impact of using majority voting and assigning the "sexist" label if at least one annotator marked it as such.

#### Final project phase 
Projects questions and challenges that we tackled in this phase: 
- experimenting with different BERT-based models
- experimenting with different autoregressive models (e.g. LLAMA)
- making the classifier's decisions explainable to user

All relevant code can be found in the `transformer_models` folder, which includes:

1. `bert_models.py` - contains the `BERTModel` class, which provides methods for initializing, fine-tuning, testing and generating explanations (using SHAP analysis) for decisions of different BERT-based models: DeBERTa, RoBERTa, HateBERT and DistilBERT. Pre-trained models and their tokenizers are sourced from the Hugging Face model repository.
2. `challenges.ipynb` - a notebook documenting various experiments with BERT-based models.
    - Firstly, we tested 4 different BERT-based models using our custom PyTorch fine-tuning implementation, as the Hugging Face Trainer API proved more time-consuming.
    - We then continued to further investigate HateBERT's potential for improvement by training in 6 different scenarios: using combinations of dataset types (balanced/unbalanced) and label aggregation methods (original labels, majority voting, at least one sexist) (section: `Different aggregation methods`). HateBERT trained on the balanced dataset with "at least one sexist" labels emerged as the best-performing model in this phase.
    - The section `Qualitative analysis of HateBERT model` conducts a SHAP analysis on custom-crafted sentences, aiming to provide us an intuition into how HateBERT makes its decisions and address the challenge of making its predictions transparent and understandable for the end-user.
    - Finally, in the section `HateBERT vs. RNN`, we performed a comparative analysis of the strengths and weaknesses of these 2 models, both quantitatively and qualitatively.

The following notebooks contain the same structure which is as follows: After importing the training data, we evaluate the base LLM
without fine-tuning on the validation dataset. Next up is the fine-tuning, which is done on both the balanced (which is called train
in the notebook) and unbalanced data set for 2-5 epochs, depending on the model. Afterward all models are evaluated for each epoch
and the best performing one regarding recall was picked for further analysis. This analysis included an evaluation on the test set,
a code cell to evaluate any comment and get the "confidence" in its prediction, i.e. the probability of the next token, an evaluation
for custom sentences, a visualization of the attention for each word and a prompt that can be used to ask the model for an explanation
for the label it has given.

3. `classification_llama3.1.ipynb`
4. `classification_llama3.2.ipynb` 
5. `classification_phi-4.ipynb`

#### Major results


Results of the best models from each section on the test set:

| Model     | Training Set Type | Label Aggregation Type | Accuracy | Balanced Accuracy | Precision | Recall |
|-----------|-------------------|------------------------|----------|-------------------|-----------|--------|
| LSTM      | Balanced          | At least one sexist    | 0.743    | 0.713             | 0.682     | 0.593  |
| HateBERT  | Balanced          | At least one sexist    | 0.781    | 0.777             | 0.689     | 0.762  |
| Llama 3.2 | Unbalanced        | Default                | 0.878    | 0.859             | 0.717     | 0.822  |


The following table contains a comparison for the best performing fine-tuned Llama 3.2 (3B), Llama 3.1 (8B) and
Phi 4 (14B) model

| Model          | Training Set Type | Label Aggregation Type | Accuracy  | Balanced Accuracy | Precision | Recall    |
|----------------|-------------------|------------------------|-----------|-------------------|-----------|-----------|
| Llama 3.2 (3B) | Unbalanced        | Default                | 0.878     | 0.859             | 0.717     | 0.822     |
| Llama 3.1 (8B) | Unbalanced        | Default                | 0.868     | **0.870**         | 0.675     | **0.875** |
| Phi 4 (14B)    | Unbalanced        | Default                | **0.893** | 0.865             | **0.765** | 0.809     |
