import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy

#download required resources
nltk.download("punkt")
nltk.download("averaged_perception_tagger_eng")
nltk.download("wordnet")
nltk.download("omw-1.4")

#step 1:Example text
text="The stripped bats are hanging on their feet for best"

#step 2:Tokenization
tokens=word_tokenize(text)

#step 3: POS tagging
pos_tags=nltk.pos_tag(tokens)

#Load spacy model
nlp=spacy.load("en_core_web_sm")

#covert NLTK POS tags to spaCy format
doc=nlp(text)
spacy_pos_tags=[(token.text,token.pos_)for token in doc]

# function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN #default to noun if unknown

#step 4:Lemmatization with POS
lemmatizer=WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token,get_wordnet_pos(pos))for token,pos in pos_tags]

#step 5:Lemmatization with spacy
spacy_lemmas = [token.lemma_ for token in doc]

#output
print("Original Tokens:",tokens)
print("NLTK POS tags:",pos_tags)
print("spaCy POS tags:",spacy_pos_tags)
print("Lemmatized TOkens using NLTK: ",lemmatized_tokens)
print("Lemmatized TOkens using spaCy: ",spacy_lemmas)

