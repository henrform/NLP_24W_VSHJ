import re
import emoji
import stanza
import nltk
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stopwords_set = set(stopwords.words('english'))
stopwords_set = stopwords_set - {'ain', 'are', 'aren', "aren't", 'can', 'couldn', "couldn't", 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'don', "don't", 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'here', 'how', 'i', 'is', 'isn', "isn't", 'just', 'me', 'mightn', "mightn't", 'more', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'only', 'ours', 'ourselves', 'shan', "shan't", 'should', "should've", 'shouldn', "shouldn't", 'they', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'who', 'whom', 'why', 'will', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've"}

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

    print("Hello")
    sentence = emoji.demojize(sentence)
    sentence = re.sub(r'\[[A-Z]+\]', '', sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s!?]+', '', sentence)

    doc = nlp(sentence)
    lemmatized_tokens = [word.lemma for s in doc.sentences for word in s.words]

    filtered_tokens = [t for t in lemmatized_tokens if t not in stopwords_set]
    return ' '.join(filtered_tokens)
