#from keras.models import Input, Model
#from keras.layers import Dense
import numpy as np
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def dsigmoid(sigmoid):
    return sigmoid*(1-sigmoid)

def getXYcbow(sentences, window, word_dict, n):
    X = []
    Y = []

    #create a sentence one-hot matrix with zero paddings
    for sentence in tqdm(sentences):
        tempVect = np.zeros((len(sentence)+2*window, n))

        for i, word in enumerate(sentence):
            tempVect[i+(window),  word_dict.get(word)] = 1
        
        for i in range(len(sentence)):
            # Getting the indices
            Y.append(tempVect[i+window, :])
            X_row= np.sum(tempVect[np.r_[i:i+window, (i+window+1):(i+2*window+1)]] ,axis=0)
            X.append(X_row)

    return X, Y

        

def cbow(sentences, vocabulary, window, d):
    """This method trains CBOW algorithm based on the inputs
        sentences: list of sentences
        window: number of words considered before and after the current word ex.5
        d: dimension of the embedding vectors
    """
    
    inp = Input(shape=(X.shape[1],))
    x = Dense(units=embed_size, activation='linear')(inp)
    x = Dense(units=Y.shape[1], activation='softmax')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')




#dummy test
sentences = [['a','b','c', 'd', 'e', 'f']] 
window=2
word_dict={'a':0,
           'b':1,
           'c':2,
           'd':3,
           'e':4,
           'f':5}
n=6

#getXYcbow(sentences, window, word_dict, n)

#see: https://github.com/MirunaPislar/Word2vec

