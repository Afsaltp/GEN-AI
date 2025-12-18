import numpy as np,nltk,pickle,tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")

corpus="I love Machine learning.I love deep learning. Machine learning is funn. Deep learning is powerful. I enjoy leaarning new things"
sentances=sent_tokenize(corpus)

tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentances)
total_words=len(tokenizer.word_index)+1
input_sequence=[]
print("Sentances:",sentances)
for line in sentances:
    print("\nProcessing line:",line)
    token_list=tokenizer.texts_to_sequences([line])[0]#convert text to sequence of token examples
    print("token List:",token_list)
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequence.append(n_gram_sequence)
        print("ngram sequence:",n_gram_sequence)
        print("current input sequence :",input_sequence)
    print("input sequence:",input_sequence)

print()
max_sequence_len=max([len(x) for x in input_sequence])
input_sequence=pad_sequences(input_sequence,maxlen=max_sequence_len,padding="pre")
print("Input Sequence:",input_sequence)

x=input_sequence[:,:-1]
print(("X:",x))
y=tf.keras.utils.to_categorical(input_sequence[:,-1],num_classes=total_words)
print("Y:",y)

#building RNN model

model=Sequential([
    Embedding(total_words,10,input_length=max_sequence_len-1),
    SimpleRNN(100),
    Dense(total_words,activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=200,verbose=1)

model.save('next_word_model.h5')

with open("tokenizer.pkl",'wb') as f:
    pickle.dump((tokenizer,max_sequence_len),f)
print("Model & Tokenizer Saved Successfully")


#Load the model
loaded_model=load_model('next_word_model.h5')
with open('tokenizer.pkl','rb') as f:
    loaded_tookenizer,loaded_max_sequence_len=pickle.load(f)

#prediction function

def predict_next(seed_text):
    token_list=loaded_tookenizer.texts_to_sequences([seed_text])[0]
    token_list=pad_sequences([token_list],maxlen=loaded_max_sequence_len-1,padding='pre')
    predicted=np.argmax(loaded_model.predict(token_list),axis=-1)

    for word,index in loaded_tookenizer.word_index.items():
        if index==predicted:
            return f"{seed_text}  --> {word}"
    return "no prediction found"

print(predict_next("I love"))
print(predict_next("deep"))
print(predict_next("machine learnig"))