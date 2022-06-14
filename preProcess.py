import nltk
import os
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer

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

    # Stemming the words

    stem_words = []
    for w in string:
        stem_words.append(stemmer.stem(w))

    return string




clean_txt = clean_text(raw_text)

token_list = nltk.word_tokenize(clean_txt)
print(token_list[20:60],"\n")
print("Total tokens : ", len(token_list))


