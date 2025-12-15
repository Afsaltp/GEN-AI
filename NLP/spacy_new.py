import spacy
nlp=spacy.load("en_core_web_sm")
doc=nlp("The Taj Mahal was build from 1887 to 1889 by French" \
        "engineer Vivek Pandey, whose company specialized in building metal frame work")
for ent in doc.ents:
    print(ent.text, ent.label_)