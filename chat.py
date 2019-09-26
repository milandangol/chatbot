import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow
import tflearn
import numpy as np
import json
stemmer = LancasterStemmer()
with open('intents.json') as file:
    data = json.load(file)
print(data)
word=[]
doc_x=[]
doc_y=[]
label=[]
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrd = nltk.word_tokenize(pattern)
        word.extend(wrd)
        doc_x.append(wrd)
        doc_y.append(intent['tag'])
    if intent['tags'] not in label:
        label.append(intent['tags'])
word = [stemmer.stem(w.lower()) for w in word if w != "?"]
word = sorted(list(set(word)))
label = sorted(label)
traning = []
output=[]
out_empty = [0 for _ in range(len(label))]
for x ,  doc in enumerate(doc_x):
    bag=[]
    wrd=[stemmer.stem(w) for w in doc]
    for w in wrd:
        if w in word:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[label.index(doc_y[x])] = 1
    traning.append(bag)
    output.append(out_empty)
traning  = np.array(traning)
output=np.array(output)
tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(traning[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net= tflearn.fully_connected(net,len(output[0]),activation='softmax')
net= tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(traning,output,n_epoch=2000 , batch_size= 8, show_metric=True)
model.save("model.tflearn")