from keras.models import Input, Model
from keras.layers import Dense, Embedding, Lambda
import keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def dsigmoid(sigmoid):
    return sigmoid*(1-sigmoid)

def getXYcbow(sentences, window, word_dict, n):
    X = []
    Y = []

    #create a sentence one-hot matrix with zero paddings
    for sentence in tqdm(sentences):
        tempVect = np.zeros((len(sentence), n))

        for i, word in enumerate(sentence):
            tempVect[i,  word_dict.get(word)] = 1
        
        for i in range(window,len(sentence)-window):
            # Getting the indices
            Y.append(tempVect[i, :])
            X_row= tempVect[np.r_[i-window:i, (i+1):(i+window+1)]]
            X.append(X_row)

    return X, Y

def getXYskipGram(sentences, window, word_dict):
    
    word_pairs = []

    for sentence in sentences:
    
        # Creating a dictrionary for context
        for i, word in enumerate(sentence):
            for w in range(window):
                
                if i + w + 1 < len(sentence): 
                    word_pairs.append([word] + [sentence[(i + 1 + w)]])
                    
                if i - w - 1 >= 0:
                    word_pairs.append([word] + [sentence[(i - w - 1)]])
                    
                    
    n_words = len(word_dict) 

    words = list(word_dict.keys())
    
    #Create X and Y by one hot encoding
    X = []
    Y = []
    
    for i, word_pair in enumerate(word_pairs):
        # Getting the indices
        focus_word_i = word_dict.get(word_pair[0])
        context_word_i = word_dict.get(word_pair[1])
    
         
        X_r = np.zeros(n_words)
        Y_r = np.zeros(n_words)
    
        #The focus word encoding
        X_r[focus_word_i] = 1
    
        #The contexts word encoding 
        Y_r[context_word_i] = 1
    
        # Appending to the main matrices
        X.append(X_r)
        Y.append(Y_r)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X,Y

def trainModel(sentences, word_dict, window, d, epochs, cbow:True):
    """This method trains CBOW algorithm based on the inputs
        sentences: list of sentences
        word_dict: dictionary of one hot indices
        window: number of words considered before and after the current word ex.5
        d: dimension of the embedding vectors
    """
    n = len(word_dict)
    if cbow:
        X,Y = getXYcbow(sentences,window,word_dict,n)
        print(np.shape(X))
        X = np.asarray(X)
        Y = np.asarray(Y)

        inp = Input(shape=(2*window,np.shape(X)[2],))
        x = Dense(units=d, activation='linear')(inp) #input_dim=np.shape(X)[1], output_dim=d, input_length=window*2)(inp)#units=d, activation='linear')(inp) #'linear'
        x = Lambda(lambda t: K.mean(t, axis=1), output_shape=(d,))(x)
        x = Dense(units=np.shape(Y)[1], activation='softmax')(x)
        model = Model(inputs=inp, outputs=x)
        opt = Adam(learning_rate=0.05)
        model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
    else:
        X,Y = getXYskipGram(sentences,window,word_dict)
        X = np.asarray(X)
        Y = np.asarray(Y)

        inp = Input(shape=(np.shape(X)[1],))
        x = Dense(units=d, activation='linear')(inp) #'linear'
        x = Dense(units=np.shape(Y)[1], activation='softmax')(x)
        model = Model(inputs=inp, outputs=x)
        opt = Adam(learning_rate=0.05)
        model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

    # Optimizing the network weights
    history = model.fit(
        x=X, 
        y=Y, 
        #validation_split=0.3,
        batch_size=258, 
        epochs=epochs,
        verbose=1
        )
    plt.figure()
    plt.plot(history.history['loss'])
    #plt.plot(history.history[''])
    #plt.legend(['train', 'test'], loc='upper left')
    plt.title('model loss')
    plt.ylabel('categorical crossentropy')
    plt.xlabel('epoch')
    if cbow:
        plt.savefig('plots/trainingCBOW.png')
    else:
        plt.savefig('plots/trainingSkip.png')
    if cbow:
        model.save('CBOWmodel.h5')
    else: 
        model.save('Skipmodel.h5')
    # Obtaining the weights from the neural network. 
    # These are the so called word embeddings

    # The input layer 
    if cbow:
        print(np.shape(model.get_weights()[0]))
        print(np.shape(model.get_weights()[1]))
        print(np.shape(model.get_weights()[2]))
        weights = model.get_weights()[0]
        print(np.shape(weights))
        embedding_dict = {}
        for word in list(word_dict.keys()): 
            embedding_dict.update({
                word: weights[word_dict.get(word),:]
                })
    else:
        weights = model.get_weights()[0]

        # Creating a dictionary to store the embeddings in. The key is a unique word and 
        # the value is the numeric vector
        embedding_dict = {}
        for word in list(word_dict.keys()): 
            embedding_dict.update({
                word: weights[word_dict.get(word)]
                })

    return embedding_dict




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

