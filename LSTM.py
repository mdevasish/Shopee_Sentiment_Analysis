# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:26:16 2021

@author: mdevasish
"""
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding,Dropout,Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
embedding_vector_features=100
ps = PorterStemmer()
voc_size = 10000
stopwords = ['she','the','is','to','of','a','i','for','in','it','you','with','that','this','was','so','will','be','my',
             'his','as','are','its','cp','have','from','there','tp','her','on','ga','at','by','if','has','me','d','does','he',
             'jd','we','your','they','here',"i'm",'aja','an','let','u','me','am','their']
sent_len = 80

mappings = {'cepet':'fast','pesen':'order','dah':'bye','lg':'again','ung':'rotten','nyampe':'arrived','yg':'which','cuman':'only',
'klo':'if','packingnya':'packing','gpp':'no problem','thx':'thanks','dapet':'get it','krn':'because','baguss':'good','gan':'bro',
'dateng':'come','pas':'just right','nyesel':'sorry','mksh':'thank you','trimakasih':'thank you','bgus':'great','smoga':'i hope',
'naman':'name','kirain':'i think','sya':'yes','pokoknya':'anyway','kok':'really','mantul':'really good','bangett':'really',
'makasi':'thanks','dong':'please','sellernya':'seller','gak':'no','uda':'already','bangettt':'already','tetep':'still','pesenan':'order',
'mudah2an':'i hope','smpai':'till','mah':'expensive','lgi':'again','lbh':'more','bagussss':'good','mantab':'steady','sukaa':'like',
'jga':'also','bnget':'really','kaka':'written','meletot':'erupted','rdtanya' : 'he asked','chargernya':'charger','dsni':'here',
'originalya':'original','jg':'too','deh':'okay','sdh':'already','tpi':'but','pokonya':'anyway','lah':'','reallyt':'really',
'sma':'school','orderan':'orders','wrna':'color','againi':'again','kak':'sis','rekomended':'recommended','kayak':'like',
'blanja':'spend','likea':'like','becausea':'because','dlu':'previous','tau':'know','barang':'goods','dtng':'come','datang':'come',
'bnyk': 'a lot','mantep':'awesome','swhich':'which','banget':'really','goodssss':'goods','rada':'rather','packagingnya':'packaging',
'skrg':'now','pngiriman':'delivery','goodsss':'goods','prev':'previous','kan':'right','kek':'grandpa','lahh':'','la':'','engga':'no',
'makasih':'thank you','ordernya ':'orders','paketan':'package','mantapp':'really','ngecewain':'disappointed','pengirimanya':'sender',
'bagus':'nice','comenya':'come','segini':'this much','knp':'why','bener':'right','kasi':'give','anak':'child','baik':'good','sukaa':'like',
'likeaaa':'like','bangeet':'really','brgnya':'how come','ngk':'presume','lagi':'again','lagii':'again','hrg':'price','harga':'price',
'penyok':'dent','penyok2':'dent','barangny':'goods','thanksi':'thanks','produk':'product','likeaaaaa':'like','murahhh':'cheap',
'terimaksih':'thanks','ownernya':'owner','thankss':'thanks','gercep':'speed','casenya':'case','kakk':'sis','dteng':'come',
'puasssss':'satisfied','masi':'still','sekali':'once','gapernah':'never','balikin':'return it','ancur':'broken','nyobain':'try it',
'bangat':'really','nyangka':'suspect','sekalii':'once','sekaliii':'once','sekaliiii':'once','sampaii':'arrive','barangnyaa':'goods',
'jaitannya':'linkage','nyari':'looking for it','bangeett':'really','disiniii':'here','abcdefghijklmnopqrstuvwxyz':'','priduk':'product',
'baguuuuuuus':'nice','allhamdulilah':'','mantaffffffffffffffff':'excellent','sekaliiiii':'once','ambilis':'take it','parahhh':'severe','Ingkan':'want'}

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def sub(df,mappings):
    df['review'] = df['review'].apply(lambda x : replace_all(x,mappings))
    return df

def process_data(df,stopwords):
    corpus = []
    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df['review'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps.stem(word) for word in review if not word in stopwords]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

def read_files():
    df_train = pd.read_csv('./Data/train.csv')
    df_test = pd.read_csv('./Data/test.csv')
    df_train.drop(['review_id'],axis = 1,inplace=True)
    df_train['rating'] = df_train['rating']-1
    df_train['rating'] = df_train['rating'].astype('category')
    return df_train,df_test

df_train,df_test = read_files()
y = df_train['rating']
df_train = sub(df_train,mappings)
df_test = sub(df_test,mappings)
df_train = process_data(df_train,stopwords)
df_test = process_data(df_test,stopwords)

onehot_repr_train=[one_hot(words,voc_size)for words in df_train] 
onehot_repr_test=[one_hot(words,voc_size)for words in df_test] 

embedded_train=pad_sequences(onehot_repr_train,padding='pre',maxlen=sent_len)
embedded_test=pad_sequences(onehot_repr_test,padding='pre',maxlen=sent_len)

tf.keras.backend.clear_session()
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_len))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(5,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(np.array(embedded_train), y, test_size=0.1,stratify = y, random_state=2021)

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64,callbacks = [es])

