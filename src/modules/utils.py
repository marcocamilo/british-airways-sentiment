import re
from bs4 import BeautifulSoup
from unidecode import unidecode
import contractions
from gensim.parsing.preprocessing import remove_stopwords

def preprocessing(text):
    # HTML decoding
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    # Special characters removal
    cleaned_text = unidecode(cleaned_text)
    # Expand contractions
    cleaned_text = contractions.fix(cleaned_text)
    # Punctuation removal
    cleaned_text = re.sub('[^\w\s]', '', cleaned_text)
    # Line breaks removal
    cleaned_text = re.sub('\n', ' ', cleaned_text)
    # Excessive spacing removal
    cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()
    # URL removal
    cleaned_text = re.sub('http\S+', '', cleaned_text)
    # Remove stopwords
    cleaned_text = remove_stopwords(cleaned_text)
    # Lowercase conversion
    cleaned_text = cleaned_text.lower()

    return cleaned_text

