import gensim.downloader as api

print("Loading pre-trained word2Vec model(Google news 3000).....")
model=api.load("word2vec-google-news-300")
print("Model loaded successfully")

# Get word vector

word = "King"
if word in model:
    vector = model[word]
    print(f"Vector for'{word}'(first 10 values): {vector[:10]}")
    print(f"Vector size: {len(vector)}\n")
else:
    print(f"'{word}' not in vocabulary\n")