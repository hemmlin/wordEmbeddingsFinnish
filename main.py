
import numpy as np
from preProcess import getTextAndVocab
from methods import traincbow
import matplotlib.pyplot as plt

def main():
    window = 2
    d=2
    sentences, word_dict = getTextAndVocab(stemming=True)
    embedding_dict= traincbow(sentences, word_dict, window, d)

    counts= np.zeros(len(word_dict))
    for sent in sentences:
        for item in sent:
            counts[word_dict.get(item)]+=1
    
    ind = np.argpartition(counts, -20)[-20:]
    
    print([counts[i] for i in ind])
    top20words= [list(word_dict.keys())[i] for i in ind]
    plt.figure(figsize=(10, 10))
    
    for word in top20words:
        coord = embedding_dict.get(word)
        plt.scatter(coord[0], coord[1])
        plt.annotate(word, (coord[0], coord[1]))
    plt.savefig('my_plot.png')
    



if __name__ == '__main__':
    main()
