
import numpy as np
from preProcess import getTextAndVocab
from methods import trainModel
import matplotlib.pyplot as plt

def getKeyByValue(dict, val):
    return [k for k, v in dict.items() if v==val]

def main():
    window = 2
    d=2
    sentences, word_dict = getTextAndVocab(stemming=False)
    embedding_dict= trainModel(sentences, word_dict, window, d, cbow=True)

    counts= np.zeros(len(word_dict))
    for sent in sentences:
        for item in sent:
            counts[word_dict.get(item)]+=1
    
    ind = np.argpartition(counts, -100)[-100:]
    
    #print(zip([getKeyByValue(word_dict,i) for i in ind] , [counts[i] for i in ind]))
    print([getKeyByValue(word_dict,i) for i in ind])

    top20words= [list(word_dict.keys())[i] for i in ind]
    plt.figure(figsize=(20, 10))
    
    for word in top20words:
        coord = embedding_dict.get(word)
        plt.scatter(coord[0], coord[1])
        plt.annotate(word, (coord[0], coord[1]))
    plt.savefig('plots/cbow.png')

    #SKIPGRAM 
    embedding_dict= trainModel(sentences, word_dict, window, d, cbow=False)

    counts= np.zeros(len(word_dict))
    for sent in sentences:
        for item in sent:
            counts[word_dict.get(item)]+=1
    
    ind = np.argpartition(counts, -100)[-100:]
    
    #print(zip([getKeyByValue(word_dict,i) for i in ind] , [counts[i] for i in ind]))
    print([getKeyByValue(word_dict,i) for i in ind])

    top20words= [list(word_dict.keys())[i] for i in ind]
    plt.close()
    plt.figure(figsize=(20, 10))
    
    for word in top20words:
        coord = embedding_dict.get(word)
        plt.scatter(coord[0], coord[1])
        plt.annotate(word, (coord[0], coord[1]))
    plt.savefig('plots/skipgram.png')
    



if __name__ == '__main__':
    main()
