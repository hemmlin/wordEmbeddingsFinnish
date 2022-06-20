from lib2to3.pgen2 import grammar
import numpy as np
from preProcess import getTextAndVocab
from methods import trainModel
from main import findNeighbours, getKeyByValue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from tensorflow import keras
import itertools
from typing import Tuple
import random
import csv
import scipy.stats


def inString(string, match):
    string_list = string#.split()
    match_list = []
    for word in string_list:
        if match in word:
            match_list.append(word)
    return match_list      

def classifyDictionary(vocabulary):
    filosofia = []
    likefilosofia = []
    karhu = []
    likekarhu = []
    tyyny = []
    liketyyny =[]
    pasta = []
    likepasta = []
    lentokone = []
    likelentokone = []
    wikipedia = []
    likewikipedia = []

    aihesanat=[]
    #aihesanat.append(inString(vocabulary,'karh'))
    #aihesanat.append(inString(vocabulary,'filo'))
    #aihesanat.append(inString(vocabulary,'tyyn'))
    #aihesanat.append(inString(vocabulary,'past'))
    #aihesanat.append(inString(vocabulary,'lento'))
    #aihesanat.append(inString(vocabulary,'wiki'))
    #print(np.shape(aihesanat))
    for word in vocabulary:#list(itertools.chain(*vocabulary)):
        luokka = input(word+ '  ')
        if luokka =='ff':
            filosofia.append(word)
        elif luokka =='f':
            likefilosofia.append(word)
        elif luokka =='kk':
            karhu.append(word)
        elif luokka == 'k':
            likekarhu.append(word)
        elif luokka =='tt':
            tyyny.append(word)
        elif luokka == 't':
            liketyyny.append(word)
        elif luokka =='ll':
            lentokone.append(word)
        elif luokka == 'l':
            likelentokone.append(word)
        elif luokka =='pp':
            pasta.append(word)
        elif luokka == 'p':
            likepasta.append(word)
        elif luokka =='ww':
            wikipedia.append(word)
        elif luokka == 'w':
            likewikipedia.append(word)

    folder='sanaluokkia/'
    #np.savetxt(folder+'filosofia.csv', np.asarray(filosofia), delimiter=',', fmt='%s')
    #np.savetxt(folder+'likefilosofia.csv', np.asarray(likefilosofia), delimiter=',', fmt='%s')
    #np.savetxt(folder+'karhu.csv', np.asarray(karhu), delimiter=',', fmt='%s')
    #np.savetxt(folder+'likekarhu.csv', np.asarray(likekarhu), delimiter=',', fmt='%s')
    #np.savetxt(folder+'tyyny.csv', np.asarray(tyyny), delimiter=',', fmt='%s')
    #np.savetxt(folder+'liketyyny.csv', np.asarray(liketyyny), delimiter=',', fmt='%s')
    #np.savetxt(folder+'pasta.csv', np.asarray(pasta), delimiter=',', fmt='%s')
    #np.savetxt(folder+'likepasta.csv', np.asarray(likepasta), delimiter=',', fmt='%s')
    #np.savetxt(folder+'lentokone.csv', np.asarray(lentokone), delimiter=',', fmt='%s')
    #np.savetxt(folder+'likelentokone.csv', np.asarray(likelentokone), delimiter=',', fmt='%s')
    #np.savetxt(folder+'wikipedia.csv', np.asarray(wikipedia), delimiter=',', fmt='%s')
    #np.savetxt(folder+'likewikipedia.csv', np.asarray(likewikipedia), delimiter=',', fmt='%s')


def uniqueCombBetween(list_1, list_2):
    unique_combinations = []

    permut = itertools.permutations(list_1, len(list_2))

    for comb in permut:
        zipped = zip(comb, list_2)
        unique_combinations.append(list(zipped))
    

    return unique_combinations

def meanDistanceIn(group, embedding_dict):
    dist=[]
    if len(group)< 100:
        #print(uniqueCombBetween(group,group)[0])
        for i, j in itertools.combinations(group,2):#uniqueCombBetween(group,group)[0]:
            dist.append(np.linalg.norm(embedding_dict.get(i)-embedding_dict.get(j)))
    else:
        for (i, j) in zip(random.choices(group,k=200),random.choices(group,k=400)):
            dist.append(np.linalg.norm(embedding_dict.get(i)-embedding_dict.get(j)))
    return np.mean(dist), np.var(dist), len(dist)



sentences, word_dict = getTextAndVocab(stemming=False)

#classifyDictionary(list(word_dict.keys())) 
counts= np.zeros(len(word_dict))
for sent in sentences:
    for item in sent:
        counts[word_dict.get(item)]+=1

plt.plot(np.arange(len(counts)), sorted(counts,reverse=True))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log(rank)')
plt.ylabel('log(frequency)')
plt.grid()
plt.savefig('plots/zipS.png')




ind = np.argpartition(counts, -10)[-10:]

#print([getKeyByValue(word_dict,i) for i in np.argwhere(counts>1)])

print( [counts[i] for i in ind])
print([getKeyByValue(word_dict,i) for i in ind])


top20words= [list(word_dict.keys())[i] for i in ind]

model = keras.models.load_model('CBOWmodelN2.h5')
weights = model.get_weights()[0]
#print(np.shape(weights))
embedding_dict_cbow = {}
for word in list(word_dict.keys()): 
    embedding_dict_cbow.update({
        word: weights[word_dict.get(word),:]
        })

model = keras.models.load_model('SkipmodelN2.h5')
weights = model.get_weights()[0]
#print(np.shape(weights))
embedding_dict_skip = {}
for word in list(word_dict.keys()): 
    embedding_dict_skip.update({
        word: weights[word_dict.get(word)]
        })


def twoSampleTtest(variable, cbow):
    data = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+variable+'.csv', dtype=str,delimiter='\n') ))))
    
    if cbow:
        mean1,variance1, n1 = meanDistanceIn(list(word_dict.keys()), embedding_dict_cbow)
        mean2,variance2, n2 =  meanDistanceIn(data, embedding_dict_cbow)
        print(variable+' mean dist CBOW: ' + str(mean2 ))
    else:
        mean1,variance1, n1 = meanDistanceIn(list(word_dict.keys()), embedding_dict_skip)
        mean2,variance2, n2 =  meanDistanceIn(data, embedding_dict_skip)
        print(variable+' mean dist skip: ' + str(mean2 ))

    sp = np.sqrt(((n1-1)*variance1+ (n2-1)*variance2)/(n1+n2-2))
    t = (mean1-mean2)/(sp*np.sqrt((1/n1)+(1/n2)))
    print('t test value for ' + variable +': ' +str(t)+'\n')
    p = scipy.stats.t.sf(abs(t), df=n1 + n2-2)
    print('p test value for ' + variable +': ' +str(p)+'\n')
    


folder='sanaluokkia/'
CBOWmean=meanDistanceIn(list(word_dict.keys()), embedding_dict_cbow)
print('CBOW mean dist: '+ str(CBOWmean))
skipmean=meanDistanceIn(list(word_dict.keys()), embedding_dict_skip)
print('Skip mean dist: '+ str(skipmean))

twoSampleTtest('filosofia',True)
twoSampleTtest('filosofia',False)
twoSampleTtest('likefilosofia',True)
twoSampleTtest('likefilosofia',False)

twoSampleTtest('karhu',True)
twoSampleTtest('karhu',False)
twoSampleTtest('likekarhu',True)
twoSampleTtest('likekarhu',False)

twoSampleTtest('wikipedia',True)
twoSampleTtest('wikipedia',False)
twoSampleTtest('likewikipedia',True)
twoSampleTtest('likewikipedia',False)

twoSampleTtest('tyyny',True)
twoSampleTtest('tyyny',False)
twoSampleTtest('liketyyny',True)
twoSampleTtest('liketyyny',False)

twoSampleTtest('pasta',True)
twoSampleTtest('pasta',False)
twoSampleTtest('likepasta',True)
twoSampleTtest('likepasta',False)

twoSampleTtest('lentokone',True)
twoSampleTtest('lentokone',False)
#twoSampleTtest('likelentokone',True)
#twoSampleTtest('likelentokone',False)
'''

filosofia = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'filosofia.csv', dtype=str,delimiter='\n') ))))#np.genfromtxt(folder+'filosofia.csv', dtype=str, delimiter='\n')
print('Filosofia in dist CBOW ' + str( meanDistanceIn(filosofia, embedding_dict_cbow)))
filosofialike  = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'likefilosofia.csv', dtype=str,delimiter='\n') ))))
print('Filosofia like in dist CBOW ' + str(meanDistanceIn(filosofialike, embedding_dict_cbow)))

filosofia = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'filosofia.csv', dtype=str,delimiter='\n') ))))#np.genfromtxt(folder+'filosofia.csv', dtype=str, delimiter='\n')
print('Filosofia in dist skip ' + str( meanDistanceIn(filosofia, embedding_dict_skip)))
filosofialike  = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'likefilosofia.csv', dtype=str,delimiter='\n') ))))
print('Filosofia like in dist skip ' + str(meanDistanceIn(filosofialike, embedding_dict_skip)))



karhu = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'karhu.csv', dtype=str,delimiter='\n') ))))#np.genfromtxt(folder+'filosofia.csv', dtype=str, delimiter='\n')
print('Karhu in dist CBOW ' + str( meanDistanceIn(karhu, embedding_dict_cbow)))
karhulike  = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'likekarhu.csv', dtype=str,delimiter='\n') ))))
print('Karhu like in dist CBOW ' + str(meanDistanceIn(karhulike, embedding_dict_cbow)))

print('Karhu in dist skip ' + str( meanDistanceIn(karhu, embedding_dict_skip)))

print('Karhu like in dist skip' + str(meanDistanceIn(karhulike, embedding_dict_skip)))

#lentsikka

karhu = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'lentokone.csv', dtype=str,delimiter='\n') ))))#np.genfromtxt(folder+'filosofia.csv', dtype=str, delimiter='\n')
print('LK in dist CBOW ' + str( meanDistanceIn(karhu, embedding_dict_cbow)))
#karhulike  = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'likelentokone.csv', dtype=str,delimiter='\n') ))))
#print('LK like in dist CBOW ' + str(meanDistanceIn(karhulike, embedding_dict_cbow)))

print('LK in dist skip ' + str( meanDistanceIn(karhu, embedding_dict_skip)))

#print('LK like in dist skip' + str(meanDistanceIn(karhulike, embedding_dict_skip)))

#pasta

karhu = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'pasta.csv', dtype=str,delimiter='\n') ))))#np.genfromtxt(folder+'filosofia.csv', dtype=str, delimiter='\n')
print('Pasta in dist CBOW ' + str( meanDistanceIn(karhu, embedding_dict_cbow)))
karhulike  = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'likepasta.csv', dtype=str,delimiter='\n') ))))
print('Pasta like in dist CBOW ' + str(meanDistanceIn(karhulike, embedding_dict_cbow)))

print('Pasta in dist skip ' + str( meanDistanceIn(karhu, embedding_dict_skip)))

print('Pasta like in dist skip' + str(meanDistanceIn(karhulike, embedding_dict_skip)))


#tyyny

karhu = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'tyyny.csv', dtype=str,delimiter='\n') ))))#np.genfromtxt(folder+'filosofia.csv', dtype=str, delimiter='\n')
print('Tyyny in dist CBOW ' + str( meanDistanceIn(karhu, embedding_dict_cbow)))
karhulike  = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'liketyyny.csv', dtype=str,delimiter='\n') ))))
print('Tyyny like in dist CBOW ' + str(meanDistanceIn(karhulike, embedding_dict_cbow)))

print('Tyyny in dist skip ' + str( meanDistanceIn(karhu, embedding_dict_skip)))

print('Tyyny like in dist skip' + str(meanDistanceIn(karhulike, embedding_dict_skip)))

#wiki

karhu = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'wikipedia.csv', dtype=str,delimiter='\n') ))))#np.genfromtxt(folder+'filosofia.csv', dtype=str, delimiter='\n')
print('Wiki in dist CBOW ' + str( meanDistanceIn(karhu, embedding_dict_cbow)))
karhulike  = list(itertools.chain(*list(csv.reader(np.loadtxt(folder+'likewikipedia.csv', dtype=str,delimiter='\n') ))))
print('Wiki like in dist CBOW ' + str(meanDistanceIn(karhulike, embedding_dict_cbow)))

print('Wiki in dist skip ' + str( meanDistanceIn(karhu, embedding_dict_skip)))

print('Wiki like in dist skip' + str(meanDistanceIn(karhulike, embedding_dict_skip)))

'''

