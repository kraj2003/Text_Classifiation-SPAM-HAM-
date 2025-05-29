from text_classification.config.configuration import DataTransformationConfig
from text_classification.exceptions.exceptions import ClassificationException
from text_classification.logging import logging
from gensim.models import Word2Vec , KeyedVectors
import gensim.downloader as api
from text_classification.utils.common import *
import sys
import os
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
import gensim
import nltk
import numpy as np

wv = api.load('word2vec-google-news-300')

class DataTransformation:
    def __init__(self, config=DataTransformationConfig):
        self.config = config
        self.wv=wv
        


    def avg2vec(self,doc,model):
        try:        # remove out-of-vocabulary words
                    #sent = [word for word in doc if word in model.wv.index_to_key]
                    #print(sent)
            logging.info("starting loading google ews 300")
            # wv= api.load('word2vec-google-news-300')
            logging.info("google news 300 completed")
            return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)
            #or [np.zeros(len(model.wv.index_to_key))], axis=0)
        except ClassificationException as e:
            raise(e,sys)


    def transform_data(self):
        try :
            # vec_king=wv['king']
            # readind the data
            messages=pd.read_csv(self.config.data_path)
            nltk.download('wordnet')
            logging.info("downloaded wornet")
            lemmatizer=WordNetLemmatizer()
            logging.info("initialized wordnet lemmatizer")
            corpus = []
            for i in range(0, len(messages)):
                review = re.sub('[^a-zA-Z]', ' ', str(messages['message'][i]))
                review = review.lower()
                review = review.split()

                review = [lemmatizer.lemmatize(word) for word in review]
                review = ' '.join(review)
                corpus.append(review)
            logging.info("lemmetized the corpus")
            [[i,j,k] for i,j,k in zip(list(map(len,corpus)),corpus, messages['message']) if i<1]
            words=[]
            for sent in corpus:
                sent_token=sent_tokenize(sent)
                for sent in sent_token:
                    words.append(simple_preprocess(sent))
            logging.info("simple pprocess done")
            
            model=gensim.models.Word2Vec(words)
            logging.info("gensim model word2vec completed")
            X=[]
            for i in tqdm(range(len(words))):
                X.append(self.avg2vec(words[i],model)) 
            logging.info("avergae w2v completed")

            X_new=np.array(X,dtype="object")
            ## Dependent Features
            ## Output Features
            print(messages['label'].head())
            y = messages[list(map(lambda x: len(x)>0 ,corpus))]
            print(y)
            y=pd.get_dummies(y['label'])
            y=y.iloc[:,0].values
            print(y)

            ## this is the final independent features
            df=pd.DataFrame()
            for i in range(0,len(X)):
                df=pd.concat([df,pd.DataFrame(X[i].reshape(1,-1))],ignore_index=True)

            # X=df.drop(['Output'],axis=1)
            # y=df['Output']
            df['output']=y
            df['output']=df['output'].astype('int')
            df.dropna(inplace=True)
            # y=y.astype('int')

            return df 

        except Exception as e:
            raise ClassificationException(e, sys)
        
    def train_test_split(self):
        try:
            data=self.transform_data()
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

            # data=pd.read_csv(self.config.data_path)
            save_object( "final_model/preprocessor.pkl", data)

        # Split the data into training and test sets. (0.75, 0.25) split.
            train, test = train_test_split(data)

            train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

            logging.info("Splited data into training and test sets")
            logging.info(train.shape)
            logging.info(test.shape)

            print(train.shape)
            print(test.shape)
        except Exception as e:
            raise ClassificationException(e, sys)