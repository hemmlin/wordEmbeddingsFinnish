# wordEmbeddingsFinnish

In this project two word embedding methods are investigated and they are applied to a small
Finnish text data. These methods are called Skip-gram and Continuous Bag of Words (CBOW).
Both of them rely on using neural networks in word embedding construction.

For the experiment we created a text file from couple of Finnish Wikipedia pages. The titles of
the selected articles were ”Karhu” (engl. Bear), ”Tyyny” (engl. Pillow), ”Pasta” (engl. Pasta),
”Filosofia” (engl. Philosophy), ”Lentokone” (engl. Airplane) and ”Wikipedia”. The data can be found from the file wiki.txt

**main.py** contains the code for runnin the whole proces and plotting some visualizations

**preProcess.py** has the data pipeline and pre processing of the data

**methods.py** contains code for creating the models and training them

**comparasion.py** has a code for some further analysis, it uses sub vocabularies under **sanaluokkia/**. It has to be run separately from the main and requires models to be already trained and saved in .h5 format.

