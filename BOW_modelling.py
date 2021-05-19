# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:49:44 2021

@author: mdevasish
"""

import pandas as pd
import numpy as np
import re
import emot
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pickle
from nltk import word_tokenize
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from nltk.stem import PorterStemmer
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.layers import Dense

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
porter = PorterStemmer()
EMOTICONS = emot.EMOTICONS

stopwords = ['she','the','is','to','of','a','i','for','in','it','you','with','that','this','was','so','will','be','my',
             'his','as','are','its','cp','have','from','there','tp','her','on','ga','at','by','if','has','me','d','does','he',
             'jd','we','your','they','here',"i'm",'aja','an','let','u','me','am','their']

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

def count_of_caps(data):
    count = 0
    text = data.split(' ')
    for each in text:
        if each.isupper():
            count+=1
    return count

def count_of_words(data):
    cnt = 0
    for text in data.split(' '):
        cnt +=1
    return cnt

def convert_emoticons(data):
    for each in EMOTICONS:
        data = re.sub(u'('+each+')'," ".join(EMOTICONS[each].replace(",","").split()),data)        
    return data

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def stemmed(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [porter.stem(each) for each in tokens]
    return " ".join(stemmed_tokens)

def feat_engg(df_train,df_test,stem = False):
    # Removal of illegal characters
    df_train['review'] = df_train['review'].apply(lambda x: x.encode('ascii','ignore').decode('utf-8'))
    df_test['review'] = df_test['review'].apply(lambda x: x.encode('ascii','ignore').decode('utf-8'))
    
    df_train['review'] = df_train['review'].apply(lambda x: convert_emoticons(x))
    df_test['review'] = df_test['review'].apply(lambda x: convert_emoticons(x))
    
    df_train['review'] = df_train['review'].apply(lambda x: x.lower())
    df_test['review'] = df_test['review'].apply(lambda x: x.lower())
    
    df_train['cap_count'] = df_train['review'].apply(lambda x : count_of_caps(x))
    df_train['word_count'] = df_train['review'].apply(lambda x: count_of_words(x))
    df_train['len'] = df_train['review'].str.len()
    
    df_test['cap_count'] = df_test['review'].apply(lambda x : count_of_caps(x))
    df_test['word_count'] = df_test['review'].apply(lambda x: count_of_words(x))
    df_test['len'] = df_test['review'].str.len()
    
    df_train['review'] = df_train['review'].apply(lambda x : replace_all(x,mappings))
    df_test['review'] = df_test['review'].apply(lambda x : replace_all(x,mappings))
    
    if stem == True:
        df_train['review'] = df_train['review'].apply(lambda x: stemmed(x))
        df_test['review'] = df_test['review'].apply(lambda x: stemmed(x))
    
    df_train = df_train[['review','cap_count','word_count','len','rating']]
    df_test = df_test[['review','cap_count','word_count','len']]
    
    return df_train,df_test

def vectorizer_form(X_train,X_val,df_test,stopwords,vec):
    
    if vec == 'tfidf':
        vec = TfidfVectorizer(min_df = 0.01,stop_words = stopwords)
        tfidf_train = vec.fit_transform(X_train['review'])
        tfidf_val = vec.transform(X_val['review'])
        tfidf_test = vec.transform(df_test['review'].tolist())
        return tfidf_train,tfidf_val,tfidf_test
    
    elif vec == 'count':
        vec = CountVectorizer(min_df = 0.01,stop_words = stopwords)
        count_train = vec.fit_transform(X_train['review'])
        count_val = vec.transform(X_val['review'])
        count_test = vec.transform(df_test['review'].tolist())
        return count_train,count_val,count_test

def create_sparse_feats(X_train,X_val,df_test,tfidf_train,tfidf_val,tfidf_test):
    
    feat_array_train = X_train[['cap_count','word_count','len']].values
    feat_array_val = X_val[['cap_count','word_count','len']].values
    feat_array_test = df_test[['cap_count','word_count','len']].values

    sparse_feat_train = sparse.csr_matrix(feat_array_train)
    feats_train = sparse.hstack([tfidf_train,sparse_feat_train])
    
    sparse_feat_val = sparse.csr_matrix(feat_array_val)
    feats_val = sparse.hstack([tfidf_val,sparse_feat_val])
    
    sparse_feat_test = sparse.csr_matrix(feat_array_test)
    feats_test = sparse.hstack([tfidf_test,sparse_feat_test])
    
    return feats_train,feats_val,feats_test

def dl_model():
    model = tf.keras.models.Sequential()
    model.add(Dense(64, activation='relu', input_dim=152))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
    return model

def implement_algo(feats_train,feats_val,y_train,y_val,algo):
    if algo == 'LogisticRegression':
        model = LogisticRegression(max_iter = 1200)
        model.fit(feats_train,y_train)
        print('Training accuracy for ',algo,model.score(feats_train,y_train))
    elif algo == 'Naive':
        model = MultinomialNB()
        model.fit(feats_train,y_train)
        print('Training accuracy for ',algo,model.score(feats_train,y_train))
    elif algo == 'xgboost':
        model = XGBClassifier(eta = 0.25,gamma = 1,max_depth = 5)
        model.fit(feats_train,y_train)
        print('Training accuracy for ',algo,model.score(feats_train,y_train))
    elif algo == 'deep':
        model = dl_model()
        history = model.fit(feats_train.todense(),np.array(y_train),
                        epochs = 50,batch_size = 32,
                        validation_data = (feats_val.todense(),np.array(y_val)),
                        callbacks = [es])
        print('Training accuracy for ',algo,accuracy_score(np.array(y_train),np.argmax(model.predict(feats_train.todense()),axis = 1)))
        print('Validation accuracy for ',algo,accuracy_score(np.array(y_val),np.argmax(model.predict(feats_val.todense()),axis = 1)))
        print('\nClassification report for training data\n',
              classification_report(y_train,
                                    np.argmax(model.predict(feats_train.todense()),axis = 1)))
                     
        print('\nClassification report for validation data\n',
              classification_report(y_val,
                                    np.argmax(model.predict(feats_val.todense()),axis = 1)))
        return model,1
    else:
        raise Exception ('Check the spellings of the algos')
    
    print('Validation accuracy for ',algo,accuracy_score(y_val,model.predict(feats_val)))
    print('\nClassification report for training data\n',
              classification_report(y_train, model.predict(feats_train)))
    
    print('\nClassification report for validation data\n',
              classification_report(y_val, model.predict(feats_val))) 
    
    return model,0

def read_files():
    df_train = pd.read_csv('./Data/train.csv')
    df_test = pd.read_csv('./Data/test.csv')
    df_train.drop(['review_id'],axis = 1,inplace=True)
    df_train['rating'] = df_train['rating']-1
    df_train['rating'] = df_train['rating'].astype('category')
    return df_train,df_test

def final_pred(model,feats_test,df_test,flag,filename):
    print('Model is ',model,'Model results are being saved')
    if flag == 0:
        df_test['rating'] = model.predict(feats_test)
    elif flag == 1:
        df_test['rating'] = np.argmax(model.predict(feats_test.todense()),axis = 1)
    else:
        raise Exception ('Invalid flag chec the inputs')
    df_test = df_test[['review','rating']]
    df_test.to_csv(filename+'.csv',index = False)
    return None
    
def impl_predict(feats_train,feats_val,feats_test,y_train,y_val,df_test,file,algorithm):
    model,flag = implement_algo(feats_train,feats_val,y_train,y_val,algo = algorithm)
    final_pred(model,feats_test,df_test,flag,filename = file)
    return None

def main():
    df_train,df_test = read_files()
    df_train,df_test = feat_engg(df_train,df_test,stem = True)
    X_train,X_val,y_train,y_val = train_test_split(df_train.iloc[:,:-1],df_train['rating'],
                                                   test_size = 0.15,random_state = 2021)
    
    train,val,test = vectorizer_form(X_train,X_val,df_test,stopwords,vec = 'count')
    feats_train,feats_val,feats_test = create_sparse_feats(X_train,X_val,df_test,train,val,test)
    
    #impl_predict(feats_train,feats_val,feats_test,y_train,y_val,df_test,file = 'xgb_tfidf_stem',algorithm = 'xgboost')
    #impl_predict(feats_train,feats_val,feats_test,y_train,y_val,df_test,file = 'naive_tfidf_stem',algorithm = 'Naive')
    #impl_predict(feats_train,feats_val,feats_test,y_train,y_val,df_test,file = 'deep_64_64_epochs=50_stem',algorithm = 'deep')
    
    impl_predict(feats_train,feats_val,feats_test,y_train,y_val,df_test,file = 'xgb_count_stem',algorithm = 'xgboost')
    impl_predict(feats_train,feats_val,feats_test,y_train,y_val,df_test,file = 'naive_count_stem',algorithm = 'Naive')
    impl_predict(feats_train,feats_val,feats_test,y_train,y_val,df_test,file = 'deep_64_64_epochs=50_stem_count',algorithm = 'deep')
    

if __name__ == "__main__":
    main()