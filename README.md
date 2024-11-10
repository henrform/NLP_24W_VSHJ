# NLP_24W_VSHJ
Topic: Detection of Online Sexism

Students: Sofija Milicic, Henry Werner Forman, Vanja Draganic, Johannes Pfennigbauer

We are using Python 3.10. The necessary packages can be found in `requirements.txt`.

#### Milestone 1
Note: The `data` folder contains 2 datasets: one with individual labels from 3 annotators per comment and another aggregated version, where 2/3 agreements were resolved by the dataset creators. For preprocessing, we used only the aggregated dataset to avoid redundant processing. The presence of 2/3 agreements will be revisited during the error analysis in milestone 2.

A detailed exploratory analysis, including text normalization steps and explanations for the preprocessing decisions, is documented in `milestone 1/milestone_1.ipynb`. The entire preprocessing workflow, including conclusions from the analysis and conversion to the standard CoNLL-U format, is implemented in the `TextProcessingPipeline` class, located in the script `milestone 1/pipeline_connl.py`. The dataset, exported in the standard format and split into training, test, and validation sets, is available in the `milestone 1` folder.

Outlined below is a diagram of the preprocessing steps implemented in the `TextProcessingPipeline` class.
<img src="assets/preprocessing_pipeline.png" width="80%"/>

#### Milestone 2