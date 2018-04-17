#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys

try:
    from nltk.stem import *
    from nltk.stem.porter import *    
    import numpy as np
    from textblob import TextBlob as tb
except:
    print("failed at loading modules, requirs numpy,textBlod,nltk.")
    sys.exit()


class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """    
    
    print("VectorSpace init...")
    #dict-- id : stemmed document, a textblod
    documentDict = dict()
    #dict-- id : document vector, with term counts
    documentStrVectorDict = dict()
    #dict-- id : document vector, with 1 when term occurs, 0 otherwise
    documentBinVectorDict = dict()
    #dict -- term : corresponding dim of document vector
    vectorKeywordIndex=dict()

    # read english.stop    
    stopword = []
    try:
        stopword = open("english.stop","r").read().split()
    except:
        print("can't load English.stop, where is it ?")
        sys.exit()
    
        
    def __init__(self, documents=dict()):
        if(len(documents)>0):
            self.build(documents)
    
    def build(self,documents):
        """ Create the vector space for the passed document strings """
        #clean words
        #regexp = re.compile("[^\w\s]")        
        print("Cleaning...")
        for key,document in documents.items():
            #self.documentDict[key] = self.cleanStemTb(document,regexp)        
            self.documentDict[key] = self.cleanStemTb(document)        

        self.vectorKeywordIndex = self.getVectorKeywordIndex(self.documentDict.values())
        print("Creating Document Vector...")
        for key,document in self.documentDict.items(): 
            try:             
                vector = self.makeVector(document)
            except:
                print("error in makevector, item:",key)
                
            self.documentStrVectorDict[key] = vector
            # create binary doc vector
            vector[vector>0]=1
            self.documentBinVectorDict[key] = vector
            #self.documentVectors = [self.makeVector(document) for document in documents.values()]           
    
    # clean stemmed remove stopwords,return textblob    
    def cleanStemTb(self,string):
        '''clean, stem words and return textblob'''
        string = string.replace(".","")
        #if(regexp):
        #string = regexp.sub(" ",string)
        regexp2 = re.compile("[^\w\s]|_")                
        string = regexp2.sub(" ",string)
        string = re.sub("[0-9]","",string)
        #string = string.replace("\s+"," ")
        string = re.sub("\s+"," ",string)
        string = string.lower()
        #remove stopword
        stemmer = PorterStemmer()
        string = [stemmer.stem(word) for word in string.split(" ") if word not in self.stopword]
        string = " ".join(string)
        return tb(string)

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """
        #get words set:        
        #wordSet = " ".join([document.string for document in documentList])
        #wordSet = wordSet.split(" ")
        wordSet = [word for document in self.documentDict.values() for word in document.words]
        wordSet = set(wordSet)            
        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in wordSet:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  
    
    # argument type : textblob
    # return type : np.array
    def makeVector(self, document):
        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        #wordList = [word for word in document.words]
        #wordList = set(wordList)
        for word,counts in document.word_counts.items():
            vector[self.vectorKeywordIndex[word]] = counts; #Use simple Term Count Model
        return np.array(vector)

    # main utility for IR
    def search(self,keyword,method="TFCOS",tfNormalize=True):
        keyword = self.cleanStemTb(keyword)
        keywordVector = self.makeVector(keyword)
        if(method=="TFCOS"):
            return self.TFCOS(keywordVector,tfNormalize)           
        elif(method=="TFED"):
            return self.TFED(keywordVector,tfNormalize)
        elif(method=="TFIDFCOS"):
            return self.TFIDFCOS(keywordVector,tfNormalize)
        elif(method=="TFIDFED"):
            return self.TFIDFED(keywordVector,tfNormalize)
        elif(method=="RelevFeedback"):
            return self.RelevanceFeedback(keywordVector,tfNormalize)
        else: 
            print("method : {}  doesn't exist!".format(method))
            return -1            
            
    def tf(self,vector,normalize=True):
        if normalize:
            return np.log10(1+vector/sum(vector))
        else:
            return vector
    
    def idf(self):
        res = [0]*len(self.vectorKeywordIndex)
        for vector in self.documentBinVectorDict.values():
            res = np.add(res,vector)
        res = np.log10(1+np.divide(len(self.documentDict),res)) 
        return(res)
        
#    def tf(word, blob,totalWord=None):
#        if totalWord : 
#            return blob.words.count(word) / totalWord
#        else :
#            return blob.words.count(word) / sum(blob.word_counts.values())
           
    def TFCOS(self,keywordVector,tfNormalize=True):
        print("search method: TF + Cosine...")
        searchTf = self.tf(keywordVector,tfNormalize)        
        searchNorm = np.linalg.norm(searchTf)
        score = dict()
        for key,document in self.documentStrVectorDict.items():    
            doc = self.tf(document,tfNormalize)
            score[key] = np.dot(doc,keywordVector)/(np.linalg.norm(doc)*searchNorm)                
        score = sorted(score.items(),key=lambda x:x[1],reverse=True)
        self.printRes(score,"TF-IDF + Cosine Similarity")
        return score

    def TFED(self,keywordVector,tfNormalize=True):
        print("search method: TF + Euclidean Distance...")        
        searchTf = self.tf(keywordVector,tfNormalize)        
        score = dict()
        for key,document in self.documentStrVectorDict.items(): 
            #score[key]= sum(np.square(np.subtract(searchTf,self.tf(document,tfNormalize))))
            score[key] = np.linalg.norm((searchTf-self.tf(document,tfNormalize)))                
        score = sorted(score.items(),key=lambda x:x[1],reverse=True)
        self.printRes(score,"TF + Euclidean Distance")
        return score

    def TFIDFCOS(self,keywordVector,tfNormalize=True):
        print("search method: TFIDF + Cosine...")                
        idfVector = self.idf()
        searchTfIdf = self.tf(keywordVector,tfNormalize)*idfVector        
        searchNorm = np.linalg.norm(searchTfIdf)
        score = dict()
        for key,document in self.documentStrVectorDict.items():    
            doc = self.tf(document,tfNormalize)*idfVector
            score[key] = np.dot(doc,searchTfIdf)/(np.linalg.norm(doc)*searchNorm)                
        score = sorted(score.items(),key=lambda x:x[1],reverse=True)
        self.printRes(score,"TF-IDF + Cosine Similarity")

        return score             

    def TFIDFED(self,keywordVector,tfNormalize=True):
        print("search method: TFIDF + Euclidean Distance...")                        
        idfVector = self.idf()
        searchTfIdf = self.tf(keywordVector,tfNormalize)*idfVector        
        score = dict()
        for key,document in self.documentStrVectorDict.items():    
            doc = self.tf(document,tfNormalize)*idfVector
            score[key] = np.linalg.norm((searchTfIdf-doc))                 
        score = sorted(score.items(),key=lambda x:x[1],reverse=True)
        self.printRes(score,"TF-IDF + Euclidean Distance")
        return score  
    
    def RelevanceFeedback(self,keywordVector,tfNormalize=True): 
         print("search method: Relevance Feedback...")                                
         res = self.TFIDFED(keywordVector,tfNormalize=True)
         feedback = self.documentDict[res[0][0]]
         target = re.compile("^[N|V]")
         words = [word for word,pos in feedback.pos_tags if target.search(pos)]                 
         newquery = 0.5*self.makeVector(tb(" ".join(words)))+ keywordVector
         idfVector = self.idf()
         searchTfIdf = self.tf(newquery,tfNormalize)*idfVector        
         searchNorm = np.linalg.norm(searchTfIdf)
         score = dict()
         for key,document in self.documentStrVectorDict.items():    
             doc = self.tf(document,tfNormalize)*idfVector
             score[key] = np.dot(doc,searchTfIdf)/(np.linalg.norm(doc)*searchNorm)                
         score = sorted(score.items(),key=lambda x:x[1],reverse=True)
         self.printRes(score,"Relevance Feedback + TF-IDF + Cosine Similarity")
         return(score)
                                    
    def printRes(self,resDict,method):
         print("_____________________＿＿＿＿＿＿\n{} \n".format(method),)
         print("DocID".ljust(10),"Score".ljust(10))
         for i in range(5):
             print("{:<10d}{:<10f}".format(int(resDict[i][0]),resDict[i][1]))
         return 0        
        
#########################################

'''
count= 0
for key,word in vectorSpace.vectorKeywordIndex.items():
    print(key,word)
    count+=1
    if count>30: break

idf = [0]*len(vectorSpace.vectorKeywordIndex)
for vec in vectorSpace.documentBinVectorDict.values():
    idf = np.add(idf,vec)

list(vectorSpace.vectorKeywordIndex.keys())[0:40]
wwwlist = {key :value.find("www") for key,value in documents.items()}
wwwlist = [key for key,value in documents.items() if value.find("www")>0 ]

'''


    
    
    