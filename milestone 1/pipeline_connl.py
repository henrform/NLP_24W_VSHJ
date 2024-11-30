import re
import logging
import pandas as pd
import stanza
from stanza.utils.conll import CoNLL
from tqdm import tqdm
import emoji
import nltk
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize custom stopwords
stopwords_set = set(stopwords.words('english'))
stopwords_set -= {'she', 'she\'s', 'herself', 'her', 'hers', 'he', 'himself', 'him', 'his',
                  'yourself', 'yourselves', 'your', 'yours'}
logger.info("Customized stopwords list: %s", sorted(stopwords_set))

# Download and initialize Stanza English pipeline
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos,depparse')

# Load dataset
df = pd.read_csv('../data/edos_labelled_aggregated.csv')


class TextProcessingPipeline:
    """Pipeline for processing text and converting it to CoNLL-U format."""

    @staticmethod
    def replace_emojis_with_description(text):
        """Replace emojis in text with their descriptions."""
        return emoji.demojize(text)

    @staticmethod
    def remove_user_mentions_and_urls(text):
        """Remove user mentions and URLs and from text."""
        text = re.sub(r'\[[A-Z]+\]', '', text)
        return text

    @staticmethod
    def clean_text(text):
        """Clean text by converting to lowercase"""
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?\'"]+', '', text)
        return text

    @staticmethod
    def remove_stopwords(text):
        """Remove stopwords from text."""
        words = nltk.word_tokenize(text)
        filtered_words = [word for word in words if word not in stopwords_set]
        return ' '.join(filtered_words)

    def preprocess_text(self, text):
        """Apply all preprocessing steps to text."""
        text = self.replace_emojis_with_description(text)
        text = self.remove_user_mentions_and_urls(text)
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

    def sentence_to_conllu_format(self, row):
        """Convert a sentence row to CoNLL-U format."""
        text = self.preprocess_text(row['text'])
        sentence_id = row['rewire_id']
        label_sexist = row['label_sexist']

        doc = nlp(text)

        conllu_format = [f"# sent_id = {sentence_id}",
                         f"# label_sexist = {label_sexist}"]

        for sentence in CoNLL.convert_dict(doc.to_dict()):
            for token in sentence:
                conllu_format.append('\t'.join(str(field) for field in token))

        return '\n'.join(conllu_format)

    def write_to_conllu(self, df, output_file):
        """Write the dataframe to a CoNLL-U formatted file."""
        total_rows = len(df)
        with open(output_file, 'w') as f:
            for _, row in tqdm(df.iterrows(), total=total_rows, desc="Processing rows", ncols=100, leave=True):
                conllu_sentence = self.sentence_to_conllu_format(row)
                f.write(conllu_sentence + '\n\n')
        logger.info("File saved: %s", output_file)


# Instantiating the pipeline
pipeline = TextProcessingPipeline()

# Splitting the data based on 'split' column
data_agg_train = df[df['split'] == 'train'].drop('split', axis=1)
data_agg_dev = df[df['split'] == 'dev'].drop('split', axis=1)
data_agg_test = df[df['split'] == 'test'].drop('split', axis=1)

# Logging the number of samples in each split
logger.info("Number of training samples: %d", data_agg_train.shape[0])
logger.info("Number of validation samples: %d", data_agg_dev.shape[0])
logger.info("Number of test samples: %d", data_agg_test.shape[0])

# Writing each split to a separate CoNLL-U file
pipeline.write_to_conllu(data_agg_train, '../data/processed/train_sexism_dataset_conllu.conllu')
pipeline.write_to_conllu(data_agg_dev, '../data/processed/dev_sexism_dataset_conllu.conllu')
pipeline.write_to_conllu(data_agg_test, '../data/processed/test_sexism_dataset_conllu.conllu')

logger.info(
    "Preprocessed datasets saved as 'train_sexism_dataset_conllu.conllu', 'dev_sexism_dataset_conllu.conllu', and 'test_sexism_dataset_conllu.conllu'.")
