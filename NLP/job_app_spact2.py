import streamlit as st
import spacy
import re

nlp = spacy.load("en_core_web_sm")

st.title("🧑‍💼 Job Requirement Extractor")

user_text = st.text_area(
    "Enter job-related text:",
    placeholder="I am looking for a data science role in Bangalore with 2 years experience at Infosys"
)

if st.button("Process"):

    doc = nlp(user_text)

    locations = []
    organizations = []

    # spaCy entity extraction
    for ent in doc.ents:
        if ent.label_ == "GPE":
            locations.append(ent.text)
        elif ent.label_ == "ORG":
            organizations.append(ent.text)

    # 🔁 FALLBACK logic (if spaCy fails)
    if not locations:
        common_locations = ["Bangalore", "Chennai", "Delhi", "Mumbai", "Hyderabad"]
        for loc in common_locations:
            if loc.lower() in user_text.lower():
                locations.append(loc)

    if not organizations:
        common_orgs = ["Infosys", "TCS", "Wipro", "Google", "Microsoft"]
        for org in common_orgs:
            if org.lower() in user_text.lower():
                organizations.append(org)

    # Experience extraction
    experience = None
    match = re.search(r"(\d+)\s*(years?|months?)", user_text.lower())
    if match:
        experience = match.group()

    # Output
    st.subheader("🔍 Extracted Information")
    st.write("📍 Location:", locations if locations else "Not found")
    st.write("🏢 Organization:", organizations if organizations else "Not found")
    st.write("⏳ Experience:", experience if experience else "Not found")

    if locations and organizations and experience:
        rewritten = f"The user wants a job in {locations[0]} with {experience} at {organizations[0]}."
    else:
        rewritten = "Not enough information to rewrite the requirement."

    st.subheader("📝 Rewritten Requirement")
    st.write(rewritten)




