import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK setup
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

class TextPreparer():

    def __init__(self,
        remove_whitespace: bool,
        remove_punctuation: bool,
        remove_stopwords: bool,
        lemmatize: bool
    ):
        self.remove_whitespace = remove_whitespace
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by removing whitespace, punctuation, stopwords, and lemmatizing."""
        if self.remove_whitespace:
            text = " ".join(text.split())

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in STOPWORDS]

        if self.lemmatize:
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

        # Rejoin tokens 
        return " ".join(tokens)