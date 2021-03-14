#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:25:04 2021

@author: Bruno Zecchi
"""

#%%
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import math
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as nlp
import sklearn.metrics.pairwise as metrics
from itertools import compress

import spacy
nameER = spacy.load('en_core_web_sm') 
stop_words = set(stopwords.words("english"))

#%%

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))


import os
filenames = []
filenames.append(os.listdir("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/BI-articles/2013/") )
filenames.append(os.listdir("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/BI-articles/2014/") )
year = ["2013","2014"]


docs = []
for yr in [0,1]:
        
    for filname in filenames[yr]:
        filpath = "/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 3/BI-articles/" + year[yr] + "/"+filname
        text = open(filpath,"r",encoding='latin-1').read()
        text = text.replace("\r"," ").replace("\n"," ").replace("  "," ")
        docs.append(text)
    
        # tokens = word_tokenize(text)
        # result = [i for i in tokens if not i in stop_words]
        
        # sentences = sent_tokenize(text)


#%%Build TFIDF table

vectorizer = nlp.TfidfVectorizer()
vectors = vectorizer.fit_transform([doc.lower() for doc in docs])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
dfTFIDF = pd.DataFrame(denselist, columns=feature_names)
    




#%%

def runner():
    print("Q1: Which companies went bankrupt in month X of year Y?")
    print("Q2: What affects GDP?")
    print("Q3: Who is the CEO of company X?")
    global choice
    choice = input("Select Q1, Q2, or Q3: Q")
    global keywords
    global extractDocs
    global sentences
    global sentences2
    global query
    global queryV
    global enType
    global finalSent
    global finalSents
    global finaldf
    global sym
    global candidateSentences
    
    enType = ""
    
    if (choice=="1"):
        keywords = ["bankrupt","bankruptcy"]
        month = input("Month X: ")
        keywords.append(month.lower())
        year = input("Year Y: ")
        keywords.append(year.lower())
        enType = "ORG"
        
        
        
    elif(choice=="2"):
        keywords = ["gdp","rise","increase","drop","decrease","effect"]
        enType = "PERCENT"
    
        
    elif(choice=="3"):
        keywords = ["ceo"]
        company = input("Company X: ")
        keywords.append(company.lower())
        enType = "PERSON"
    
        
        
    else:
        print("Incorrect input, select Q1, Q2, or Q3. \n")
        runner()
      
    
    
    #####################
    
    
    
    
     #extract docs with only at least 1 of keywords present   
    extractDocs = []
    indice = 0
    for doc in docs:
        if any(kw in doc.lower() for kw in keywords):
            extractDocs.append(indice)
        indice += 1
        
    scoring() #run scoring function for all extracted documents
    
    #    stemmer = PorterStemmer()
     #   result_stemming = [stemmer.stem(word) for word in result]
    
     
    fullDocs = ""
    for d in selectedDocs:
        fullDocs = fullDocs + " "+ d
    
    sentences = sent_tokenize(fullDocs)
    
    if choice=="2": # sentences for question 2 MUST contain "GDP"
        sentences  = [s for s in sentences if "gdp" in s.lower()]
    if choice =="3":# sentences for question 3 MUST contain "CEO"
        sentences  = [s for s in sentences if "ceo" in s.lower()]
    sentences2 = NER(sentences)

    
    
    sentences3 = [s.lower() for s in sentences2]
    sentences3.append(" ".join(keywords))
    
    
    ############## Build TFIDF of sentences
    
    vectorizer = nlp.TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences3)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    dfSentc = pd.DataFrame(denselist, columns=feature_names)
    
    
    
    ####### Find cosine similarity of sentences
    
    sym = metrics.cosine_similarity(dfSentc.values)
    candidateSentences = sym[-1] 
    candidateSentences[candidateSentences>=1] = 0
    candidateSentences = np.array(candidateSentences) #### get similarities between query and sentences
       
    finaldf = pd.DataFrame()
    finaldf["index"] = range(len(candidateSentences))
    finaldf["sentscore"] = candidateSentences
    finaldf.sort_values(by = "sentscore",inplace=True,ascending=False)
    
    
    ### Prepare output of answer
    print("Possible answers:")
    if choice == "1":
        sentsindex = finaldf.head(10).index
        finalSents = ""
        for x in range(len(sentsindex)):
            finalSents = finalSents + " "+ sentences2[sentsindex[x]]
        doc2 = nameER(finalSents)
        answer = [e.text for e in doc2.ents if e.label_ ==enType]
        answer = set(answer)
        print(answer)
        print("\n")
        runner()
    
    elif choice =="2":
        dfgdp = pd.DataFrame()
        
        
        
        sentsindex = finaldf.head(10).index
        
        #build lists of factors and their corresponding percentages
        factors = [[]]
        percs = [[]]
        for s in range(10):
            factors.append([])
            percs.append([])
            doc3 = nameER(sentences2[sentsindex[s]])
            for d in doc3:
                if d.pos_ == "NOUN":
                    factors[s].append(str(d)) 
            for d2 in doc3.ents:
                if d2.label_ == "PERCENT":
                    percs[s].append(str(d2))
                
        totfactors = []
        for f in factors:
            for f2 in f:
                totfactors.append(str(f2))
                
        totpercs = []
        for f in percs:
            for f2 in f:
                totpercs.append(str(f2))
                
   
        totfactors = [ele for ele in totfactors if ele not in ["percent","growth","GDP","%","increase","drop","decrease","rise","effect"]]
        totfactors = list(set(totfactors))
        print(totfactors) # provide list of factors
        choice2 = input("Select factor to see percentage change in GDP associated with factor: ")
        #find percentages tied with factor chosen
        relFacts = [choice2 in f for f in factors]
        relPercs = list(compress(percs, relFacts))
        relPercs = [item for sublist in relPercs for item in sublist]
        answer = set(relPercs)
        print(answer)
        print("\n")
        runner()
        
        
        
        
        
    elif choice =="3":
        finalSent = sentences2[candidateSentences.argmax()]
        doc2 = nameER(finalSent) #if final selected sentences, remove those that don't contain entity type Person
        answer = [e.text for e in doc2.ents if e.label_ ==enType]
        answer = set(answer)
        print(answer)
        print("\n")
        runner()
    

        



#%%Find top scoring document(s)
def scoring():
    global selectedDocs
    global subTI
    selectedDocs = [[]]
    cols = dfTFIDF.columns.isin(keywords)
    subTI = dfTFIDF.iloc[extractDocs,cols]
    subTI = subTI[(subTI!=0).all(axis=1)]
    
    topvals = subTI.quantile(q=0.5)
    
    subTI = subTI[(subTI > topvals).all(axis=1)]
    subTI["suma"] = subTI.sum(axis=1)
    
    subTI.sort_values(by="suma",ascending=False,inplace=True)

    if (choice=="1" or choice=="2") and len(subTI)>1:
        mark = min(10,len(subTI))
        selectedDocs = [docs[ind] for ind in subTI.index.values[0:mark]]
    elif choice=="3" or len(subTI)==1:
        selectedDocs[0] = docs[subTI.index.values[0]]

#%% Shrink down sentences to only ones that have relevant entity type
def NER(sents):

    indice2 = 0
    candidSents = []
    for s in sents:
        doc = nameER(s)
        for e in doc.ents:
            if e.label_ == enType:
                candidSents.append(sents[indice2])
                
        indice2 += 1
    
    return candidSents

#%% Run Code. Run this cell!

runner()

