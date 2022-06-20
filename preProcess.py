import nltk
import os
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
from nltk import tokenize

#nltk.download('stopwords')
stemmer = SnowballStemmer("finnish")
finnish_stopwords = stopwords.words('finnish')

#file = open(os.getcwd()+ "/archive/data_fin_gutenberg_500.txt","rt", encoding = 'utf-8')
file = open(os.getcwd()+ "/wiki.txt","rt", encoding = 'utf-8')
raw_text = file.read(100000)

file.close()

def clean_text(
    string: str, 
    punctuations=r'''()-[]{};:'"\,<>/@#$%^&*_~»«0123456789–””''') -> str: #,
    """
    A method to clean text 
    """

    # Cleaning the urls
    string = re.sub(r'https?://\S+|www\.\S+', '', string)

    # Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)

    
    


    # Removing the punctuations
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 

     # Converting the text to lower
    string = string.lower()

    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in finnish_stopwords])

     # Cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()



    return string







def getTextAndVocab(stemming:True):
 
    clean_txt = clean_text(raw_text)

    #token_list = nltk.word_tokenize(clean_txt)
    
    token_list=[]
    
    for i in nltk.sent_tokenize(clean_txt):
        temp = []
         
        # tokenize the sentence into words
        for j in nltk.word_tokenize(i):
            if j not in ['!','.','?', '...', '..']:
                temp.append(j)
        if len(temp)>5:
            token_list.append(temp)
    
    if stemming:
        # Stemming the word
        for i, sentence in enumerate(token_list):
        
            stem_words = []
            for w in sentence:
                stem_words.append(stemmer.stem(w))
            
            token_list[i] = stem_words

    print(token_list[:30],"\n")
    print("Total tokens : ", len(token_list))

    Vocab=[]
    for sent in token_list:
        for item in sent:
            if not item in Vocab:
                Vocab.append(item)
    #print(Vocab)

    word_dict = {}

    for i, word in enumerate(Vocab):
            word_dict.update({
                word: i
            })

    print(len(Vocab))

    return (token_list ,word_dict)


