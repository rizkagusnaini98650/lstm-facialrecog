import string
import re
import nltk
import preprocessor as p
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

pattern = r'[0-9]'

def preprocessed_text(text):
    for punctuation in string.punctuation:
        text = p.clean(text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = text.replace(punctuation, '')
        text = re.sub(pattern, '', text)
        text = re.sub(r'\r?\n|\r', '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = text.lower()
        text = text.split()
        stop_words = set(stopwords.words('english'))
        text = [token for token in text if token not in stop_words]
        text = " ".join(text)
        ps = PorterStemmer()
        text = ps.stem(text)
    return text
