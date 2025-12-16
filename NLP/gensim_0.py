import gensim.downloader as api

print("Loading pre-trained word2Vec model(Google news 3000).....")
model=api.load("word2vec-google-news-300")
print("Model loaded successfully")

print("Most similar to 'king':")
print(model.most_similar("king",topn=5))

print("\nking - man + woman:")
print(model.most_similar(positive=["king","woman"],negative=["man"],topn=5))