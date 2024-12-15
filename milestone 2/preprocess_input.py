import re
import emoji
import stanza
import nltk
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stopwords_set = set(stopwords.words('english'))
stopwords_set -= {'she', 'she\'s', 'herself', 'her', 'hers', 'he', 'himself', 'him', 'his',
                  'yourself', 'yourselves', 'your', 'yours'}

stanza.download('en', verbose=False)
nlp = stanza.Pipeline('en', processors='tokenize,lemma', use_gpu=False, verbose=False)


def preprocess_single_sentence(sentence: str) -> str:
    """
    Preprocess a single sentence:
    - Replace emojis with descriptions
    - Remove user mentions and URLs
    - Clean text (lowercase, remove unwanted characters)
    - Tokenize and lemmatize with Stanza
    - Remove stopwords
    Returns a string of lemmatized tokens without stopwords.
    """

    sentence = emoji.demojize(sentence)
    sentence = re.sub(r'\[[A-Z]+\]', '', sentence)
    sentence = re.sub(r'http\S+|www.\S+', '', sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s.,!?\'"]+', '', sentence)

    doc = nlp(sentence)
    lemmatized_tokens = [word.lemma for s in doc.sentences for word in s.words]

    filtered_tokens = [t for t in lemmatized_tokens if t not in stopwords_set]
    return ' '.join(filtered_tokens)
