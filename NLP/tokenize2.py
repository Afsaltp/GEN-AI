import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize
text="Hello there! Welcome to NLP."
text="Mr.smith "
print(word_tokenize(text))
print(sent_tokenize(text))