
from turtle import pu
import numpy as np
from preProcess import getTextAndVocab
from methods import trainModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches


def getKeyByValue(dict, val):
    return [k for k, v in dict.items() if v==val]

def getColor(word):
    colorwords=['filosof','karh', 'lentokon','past', 'tyyn', 'wiki']
    colors = ['green','tab:brown', 'blue', 'orange','purple' , 'red', 'black']
    for i, match in enumerate(colorwords):
        if match in word:
            return colors[i]
    return 'black'

def plotPoints(words,dict,d,path):
    # Plots the words in the embedding space 
    # and projects them to the 2 first principa components if the dimension is >2
    green_patch = mpatches.Patch(color='green', label='philosophy')
    brown_patch = mpatches.Patch(color='tab:brown', label='bear')
    blue_patch = mpatches.Patch(color='blue', label='airplane')
    orange_patch = mpatches.Patch(color='orange', label='pasta')
    purple_patch = mpatches.Patch(color='purple', label='pillow')
    red_patch = mpatches.Patch(color='red', label='wikipedia')
    black_patch = mpatches.Patch(color='black', label='other')

    
    if d==2:
        plt.figure(figsize=(10, 10))
    
        for word in words:
            coord = dict.get(word)
            col =  getColor(word)
            plt.scatter(coord[0], coord[1], c=col)
            plt.annotate(word, (coord[0], coord[1]), fontsize=10)
    else:
        #PCA and visualization on the 2. first components
        plt.figure(figsize=(10, 10))
        pca = PCA(n_components=2)
        print(np.shape(list(dict.values())))
        principalComponents = pca.fit_transform(list(dict.values()))
        print(np.shape(principalComponents))
        dictWords=list(dict.keys())
        print('Components explain')
        pca2 = PCA()
        pca2.fit_transform(list(dict.values()))
        print(pca2.explained_variance_ratio_)
        
        for word in words:
            col =  getColor(word)
            coord = principalComponents[dictWords.index(word),:]
            plt.scatter(coord[0], coord[1], c=col)
            plt.annotate(word, (coord[0], coord[1]), fontsize=10)
            plt.xlabel('1. principal component')
            plt.ylabel('2. principal component')
        

    #plt.show()
    plt.legend(handles=[red_patch,green_patch, brown_patch, blue_patch, orange_patch, purple_patch, black_patch])
    plt.savefig(path)

def findNeighbours(word, embedding_dict, n):
    # Finds the nearest neighbours of a word in the embedding space.
    # Using euclidean distance
    topN = []
    for i in range(n):
        topN.append('')
    #print(topN)
    distN = np.zeros(n)+1000
    if word in list(embedding_dict.keys()):
        embeddingW = embedding_dict.get(word)
        maxDist=1000
        for item in list(embedding_dict.keys()):
            dist= np.linalg.norm(embeddingW-embedding_dict.get(item))
            if dist < maxDist:
                #print(np.argmax(distN, axis=0))
                topN[np.argmax(distN, axis=0)] = item
                distN[np.argmax(distN, axis=0)] = dist
                maxDist = np.max(distN)
    else:
        print('word is not in the vocabulary')
    return zip(*sorted(zip(distN, topN))) #topN[np.argsort(distN)], distN[np.argsort(distN)]
        #test



def main():
    # training parameters
    window = 2
    d= 20
    epochs=1000
    sentences, word_dict = getTextAndVocab(stemming=False)
    
    #calculating frequencies of words
    counts= np.zeros(len(word_dict))
    for sent in sentences:
        for item in sent:
            counts[word_dict.get(item)]+=1
    
    ind = np.argpartition(counts, -100)[-100:]
    
    print( [counts[i] for i in ind])
    print([getKeyByValue(word_dict,i) for i in ind])

    top20words= [list(word_dict.keys())[i] for i in ind]
    
    # training the CBOW model
    embedding_dict_cbow= trainModel(sentences, word_dict, window, d, epochs, cbow=True)

    plotPoints(top20words,embedding_dict_cbow,d,'plots/cbow.png')

    #training Skip-gram model
    embedding_dict_skip= trainModel(sentences, word_dict, window, d, epochs, cbow=False)
    
    #print(zip([getKeyByValue(word_dict,i) for i in ind] , [counts[i] for i in ind]))
    plotPoints(top20words,embedding_dict_skip,d,'plots/skipgram.png')

    print('VALMIS')
    # Testing the models by printing some nearest neighbours
    testWords= top20words[:10]#['äiti', 'isä', 'maa', 'talon', 'kyllä', 'suuri']
    for test in testWords:
        print(test)
        print("CBOW")
        for pair in findNeighbours(test, embedding_dict_cbow,10):
            print(pair)

        print("Skip")
        for pair in findNeighbours(test, embedding_dict_skip,10):
            print(pair)
    
    testWords= ['wikipedia', 'karhu', 'tyyny', 'pasta', 'lentokone', 'filosofia']
    for test in testWords:
        print(test)
        print("CBOW")
        for pair in findNeighbours(test, embedding_dict_cbow,10):
            print(pair)

        print("Skip")
        for pair in findNeighbours(test, embedding_dict_skip,10):
            print(pair)







if __name__ == '__main__':
    main()
