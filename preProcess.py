import nltk
import os
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
from nltk import tokenize

#nltk.download('stopwords')
stemmer = SnowballStemmer("finnish")
finnish_stopwords = stopwords.words('finnish')

file = open(os.getcwd()+ "/archive/data_fin_gutenberg_500.txt","rt")
raw_text = file.read(1000000)

file.close()

def clean_text(
    string: str, 
    punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''') -> str: #,
    """
    A method to clean text 
    """

    # Cleaning the urls
    string = re.sub(r'https?://\S+|www\.\S+', '', string)

    # Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)

    
    sentList = tokenize.sent_tokenize(string)
    print(sentList[:5])
    for string in sentList:


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



    return sentList



def getTextAndVocab(stemming:True):
 
    clean_txt = clean_text(raw_text)

    token_list = nltk.word_tokenize(clean_txt)

    if stemming:
        # Stemming the word
        stem_words = []
        for w in token_list:
            stem_words.append(stemmer.stem(w))
        token_list = stem_words

    print(token_list[20:60],"\n")
    print("Total tokens : ", len(token_list))

    Vocab=[]
    for item in stem_words:
        if not item in Vocab:
            Vocab.append(item)
    print(Vocab[:50])

    word_dict = {}

    for i, word in enumerate(Vocab):
            word_dict.update({
                word: i
            })

    print(len(Vocab))

    return (token_list ,word_dict)


