
import numpy as np
from preProcess import getTextAndVocab
from methods import trainModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

def getKeyByValue(dict, val):
    return [k for k, v in dict.items() if v==val]

def plotPoints(words,dict,d,path):
    if d==2:
        plt.figure(figsize=(20, 10))
    
        for word in words:
            coord = dict.get(word)
            plt.scatter(coord[0], coord[1])
            plt.annotate(word, (coord[0], coord[1]), fontsize=12)
    else:
        fig = plt.figure(dpi=60)
        ax = fig.gca(projection='3d')
        #ax.set_axis_off()
        for word in words:
            coord = dict.get(word)
            ax.scatter(coord[0], coord[1], coord[2], marker='o', c = 'c', s = 44) 
            annotate3D(ax, s=word, xyz=(coord[0], coord[1], coord[2]), fontsize=7, xytext=(-3,3),
                       textcoords='offset points', ha='right',va='bottom')     
    #plt.show()
    plt.savefig(path)

def findNeighbours(word, embedding_dict, n):
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
    window = 2
    d=5
    epochs=3000
    sentences, word_dict = getTextAndVocab(stemming=False)
    
    counts= np.zeros(len(word_dict))
    for sent in sentences:
        for item in sent:
            counts[word_dict.get(item)]+=1
    
    ind = np.argpartition(counts, -100)[-100:]
    
    print( [counts[i] for i in ind])
    print([getKeyByValue(word_dict,i) for i in ind])

    top20words= [list(word_dict.keys())[i] for i in ind]
    
    #CBOW
    embedding_dict_cbow= trainModel(sentences, word_dict, window, d, epochs, cbow=True)

    plotPoints(top20words,embedding_dict_cbow,d,'plots/cbow.png')
    #SKIPGRAM 
    embedding_dict_skip= trainModel(sentences, word_dict, window, d, epochs, cbow=False)
    
    #print(zip([getKeyByValue(word_dict,i) for i in ind] , [counts[i] for i in ind]))
    plotPoints(top20words,embedding_dict_skip,d,'plots/skipgram.png')

    print('VALMIS')
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
