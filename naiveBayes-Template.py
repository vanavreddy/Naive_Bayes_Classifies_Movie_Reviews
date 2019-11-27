#!/usr/bin/python

import sys
import os
import re
import numpy as np
import nltk
import random
from collections import Counter
nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)
nltk.download('movie_reviews', quiet = True)
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB


###############################################################################

#Function is used for extract text after cleaning up the stopwords
#Input: text from document
#Output: Cleaned text without stopwords
#Since the choice-1 requires no pre-processing, I ended up not using this for final results
'''
def extract_words(text):    
    #ignore = ['a', "the", "is"]  
    #Stopwords list generated from nltk and copied here
    stopwords = ['a', 've', 'just', "isn't", 'ours', 'only', 'such', 'mightn', 'was', 'until', 
                 'didn', 'wasn', 'about', "hadn't", 'y', 'down', 'weren', 'to', 'doesn', 
                 'any', 'themselves', 'while', 'more', 'in', 'been', "don't", 'theirs', 
                 'which', 'yours', 'but', 'once', 'o', 'had', 'there', 'isn', 'doing', 
                 'its', 'will', "you've", 'above', 's', 'have', 'all', "mightn't", 'be', 
                 "it's", 'those', 'yourselves', 'were', 'and', 'his', 'most', "wasn't", 
                 'no', 'own', 'so', "shouldn't", 'am', 'is', 'by', 'after', 'yourself', 
                 'do', "should've", 'again', 'over', "hasn't", 'shouldn', 'that', "shan't", 
                 'aren', 'if', 'has', 'herself', 'very', "didn't", "that'll", 'when', 
                 "couldn't", "haven't", 'won', 'than', 'an', 'nor', 'are', 'whom', 'under', 
                 'not', 'i', 'my', 'him', 'against', 'as', 'from', 'hers', "won't", "you're", 
                 'both', 'wouldn', 'hadn', 'who', 'some', 'or', 're', 'should', 'mustn', 
                 'below', 'm', 'during', "doesn't", 'further', 'itself', 'on', 'll', 'we', 
                 'where', 'too', 'she', 'other', "weren't", 'now', 'them', 'these', 'few', 
                 'does', 'can', 'ain', 'her', 'the', 'having', 'between', 'our', 'did', 
                 'same', 'it', 'being', 'me', "aren't", "mustn't", 'you', 'your', 'into', 
                 'up', 'why', 'each', 'couldn', "wouldn't", 'd', 'here', "you'll", "she's", 
                 'myself', 'haven', 'ourselves', 'ma', 't', 'of', 'he', 'because', 'at', 
                 'out', 'this', 'before', 'himself', 'then', "needn't", "you'd", 'for', 
                 'they', 'what', 'how', 'their', 'hasn', 'needn', 'shan', 'don', 'through', 
                 'off', 'with'] 
    
    words = re.sub("[^\w]", " ",  text).split()    
    clean_text = [w.lower() for w in words if w not in stopwords]    
    
    return clean_text
'''

#Function to find similarity between two words
#Input: Two words, word is dictionary word, the_word is a word in text read from document
#Output: similarity number between two words
def similar_text(word, the_word):
   
    similar = 0.0
    
    if word[:len(word)-1] == the_word[:len(word)-1] :
        word = word + ' ' * (len(the_word) - len(word)-1)
        the_word = the_word + ' ' * (len(word) - len(the_word))
        similar = sum(1 if i == j else 0 for i, j in zip(word, the_word)) / float(len(word))
    
    return similar

#Function is used find similar words in text, if similar words are found replace with word from dictionary
#Input: text from document
#Output: similar words replace in text with words from dictionary
def match_text(vocabulary, cleaned_text):
    
    #I came up with this list by looking and the given dictionary words
    #Basically, for each word in given dictionary, I thought of possible words and put them in list
    matching_words = ['bore', 'bored', 'super', 'loving', 'loved', 'lovely', 'like', 'wasting',
                 'wasted', 'entertained', 'visually', 'success', 'successfully', 'formerly',
                 'similarly', 'drama', 'personally', 'popularity', 'certainly', 'difficulty',
                 'happily', 'happiest', 'happier', 'worst', 'darkest', 'darker', 'easiest', 
                 'easier', 'wonderfully', 'importantly', 'wildest', 'wilder', 'seriously', 
                 'recently', 'poorer', 'poorest', 'stupidity', 'deeper', 'deepest', 'emotionally',
                 'romance', 'possibly', 'simpler', 'strongest', 'stronger', 'shortest', 'shorter',
                 'whitest', 'whiter', 'beauty', 'beautifully', 'obviously', 'perfectly', 'perfection',
                 'worse', 'majority', 'fully', 'nicely', 'nicer', 'deadly', 'comedy', 'finally', 'truely',
                 'humanity', 'mainly', 'smaller', 'smallest', 'entirely', 'surely', 'harder',
                 'hardest', 'differently', 'longer', 'longest', 'interested', 'highest', 
                 'funnier', 'funniest', 'oldest', 'oldest', 'younger', 'youngest', 'bigger',
                 'biggest', 'greater', 'greatest', 'newer', 'newest']
    
    #print("Before Matching:\n", cleaned_text)
    j = 0
    for the_word in cleaned_text:  
        for i, word in enumerate(vocabulary): 
            #print(word, the_word, i)
            if the_word not in matching_words:
                continue
            elif similar_text(word, the_word) > 0.50:
                #print("Similar: ", word, the_word)
                cleaned_text[j] = word
        j = j+1      
    #print("After Matching:\n", cleaned_text)        
    return cleaned_text

#Function to split text into words and match simiar words 
#Input: dictionary, text from document
#Output: text with similar words matched and replaced
def extract_and_match_text(vocabulary, text):
    
    #split the text into words
    words = text.split()
    #print(words)
    matched_text = match_text(vocabulary, words)
    #print(matched_text)
    
    return matched_text

#Function is used transform document into Bag of Words representation
#Input: File/document, dictionary, choice
#Output: Bag of Words from given document
def transfer(fileDj, vocabulary, choice):
    
    BOWDj = np.zeros(len(vocabulary), dtype='int64')
    
    text = []
    with open(fileDj, 'r') as file:
        text = file.read().replace('\n', '')
    
    if (choice == 1):
        #Hand engineered
        #print("Choice 1")
        
        #get similar words replaced with words from dictionary 
        matched_text = extract_and_match_text(vocabulary, text)
        
        '''
        cleaned_text = extract_words(text)
        #print(cleaned_text)
        matched_text = match_text(vocabulary, cleaned_text)
        
        print(len(matched_text))
        #print(matched_text) 
        '''
        
        found = 0
        for the_word in matched_text:  
            for i, word in enumerate(vocabulary): 
                #print(word, the_word, i)
                if word == the_word: 
                    #print("Matched: ", word, the_word)
                    BOWDj[i] += 1
                    found += 1
        #removing UKN count per post on Piazza   
        #BOWDj[-1] = len(matched_text) - found
        
        #print(np.array(BOWDj))
        
    elif (choice == 2):
        #NLTK
        stop_words = []
        #RegEx for selecting only alphanumerics chars, drops others
        tokenizer = RegexpTokenizer(r'\w+')
        #Tokenize the words
        tokenized_text = tokenizer.tokenize(text)
        #print(tokenized_text)
        #Get English stopwords
        stop_words = set(stopwords.words("english"))
        #print(stop_words)
        #Remove stop words from text
        cleaned_text = []
        for w in tokenized_text:
            if w not in stop_words:
                cleaned_text.append(w)
        #print(cleaned_text)
        #Stemming using PorterStemmer
        ps = PorterStemmer()
        stemmed_text=[]
        for w in cleaned_text:
            stemmed_text.append(ps.stem(w))
        
        #print(stemmed_text)   
        found = 0
        for the_word in stemmed_text:  
            for i, word in enumerate(vocabulary): 
                #print(word, the_word, i)
                if word == the_word:  
                    #print("Matched: ", word, the_word)
                    BOWDj[i] += 1
                    found += 1
         
        BOWDj[-1] = len(stemmed_text) - found
                             
        #print(np.array(BOWDj))
    else:
        print("Unknow Choice\n")
        exit
        
    return BOWDj

#Function read all the training and testing documents 
#Input: Absolute file system path to the data sets
#Output: Matrices Xtrain, Xtest, ytrain and ytest
def loadData(Path):
    
    vocab = []
    CHOICE = 2
    k = 2000
    
    if CHOICE == 1:
        #open dictionary and read into standard vocabulary
        dictionary = os.path.join(Path, '../dictionary.txt')
        with open(dictionary, 'r') as file:
                vocab = file.read().split()
    elif CHOICE == 2:
        #build dictionary from NLTK movie_reviews corpus
        all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
        vocab = list(all_words)[:k]
        
    #print("Vocabulary:\n", vocab)
    
    Xtrain, Xtest, ytrain, ytest = [], [], [], []
    data = []
    #Process Taining Data
    #print("\nGetting Training data")
    training_data = os.path.join(Path, 'training_set')
    for rev_type in os.listdir(training_data):

        if(rev_type == 'pos'):
            y_value = 1
        else:
            y_value = -1
            
        rev_type = os.path.join(training_data, rev_type)
        for docj in os.listdir(rev_type):
            docj = os.path.join(rev_type, docj)
            data.append([transfer(docj, vocab, CHOICE), y_value])
    
    
    for x, y in data:
        Xtrain.append(x)
        ytrain.append(y)
    
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
     
    data = []
    #Process Test Data
    #print("\nGetting Test data")
    test_data = os.path.join(Path, 'test_set')
    for rev_type in os.listdir(test_data):

        if(rev_type == 'pos'):
            y_value = 1
        else:
            y_value = -1
            
        rev_type = os.path.join(test_data, rev_type)
        for docj in os.listdir(rev_type):
            docj = os.path.join(rev_type, docj)
            data.append([transfer(docj, vocab, CHOICE), y_value])
    
    for x, y in data:
        Xtest.append(x)
        ytest.append(y)
    
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    
    return Xtrain, Xtest, ytrain, ytest

#Function for training the model
#Input: Xtrain, ytrain
#Output: thetaPos, ThetaNeg
def naiveBayesMulFeature_train(Xtrain, ytrain):
    
    a = 1 #smoothing factor
    v = Xtrain.shape[1] # no.of words in vocabulary
    n = 0 #Sum of all words in a given class, calculated below
    
    doc_num = 0
    pos_Xtrain = []
    neg_Xtrain = []
    
    #divide positive and negative cases, this may not be needed if we always assume
    #that the data will be equally distributed and sorted. But, I am generalizing it here
    #in case we get data mixed in 
    doc_num = 0
    for docj in Xtrain:
        if ytrain[doc_num] == 1:
            pos_Xtrain.append(docj)
        elif ytrain[doc_num] == -1:
            neg_Xtrain.append(docj)  
        doc_num += 1
    prior_p = len(pos_Xtrain)/Xtrain.shape[0]
    prior_n = len(neg_Xtrain)/Xtrain.shape[0]
    
    #Calculate P(Wi/cj), probability of word wi for given class cj
    ####### POSITIVE ############
    n = np.sum(pos_Xtrain)
    v_d = v + a * n
    #add all the counts of wi
    pos_Xtrain = np.sum(pos_Xtrain, axis=0)
    #add laplace smoothing
    pos_Xtrain = np.add(pos_Xtrain, a)
    #multiply by p_cj=0.5
    pos_Xtrain = np.multiply(pos_Xtrain, prior_p)
    thetaPos = np.divide(pos_Xtrain, v_d)
    
    ####### NEGATIVE ############
    n = np.sum(neg_Xtrain)
    v_d = v + a * n
    #add all the counts of wi
    neg_Xtrain = np.sum(neg_Xtrain, axis=0)
    #add laplace smoothing
    neg_Xtrain = np.add(neg_Xtrain, a)
    #multiply by p_cj=0.5
    neg_Xtrain = np.multiply(neg_Xtrain, prior_n)
    thetaNeg = np.divide(neg_Xtrain, v_d)
    
    return thetaPos, thetaNeg

#Function for testing the trained model - MNBC
#Input: Xtest, ytest
#Output: predictions, accuracy
def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    yPredict = []
    Accuracy = 0.0
    #assuming we will always have 2 classes and equal number of samples in each class
    prior = 0.5
    
    for docj in Xtest:
        prod_p = np.multiply(docj, np.log(thetaPos))
        prod_p = np.multiply(prod_p, prior)
        sum_p = np.sum(prod_p)
        #print("Sum Pos: ", sum_p)
        
        prod_n = np.multiply(docj, np.log(thetaNeg))
        prod_n = np.multiply(prod_n, prior)
        sum_n = np.sum(prod_n)
        #print("Sum Neg: ", sum_n)
        
        if sum_p > sum_n:
            yPredict.append(1)
        else:
            yPredict.append(-1)
    
    #print("Predicted Labels:" , yPredict)
    #print("Actual Labels:" , ytest)
    
    total = 0
    for i, j in zip(yPredict, ytest):
        if i == j:
            total +=1
        else:
            pass   
        
    Accuracy = total/len(ytest) * 100

    return yPredict, Accuracy

#Function for training and testing using sklearn
#Input: Xtrain, ytrain, Xtest, ytest
#Output: Accuracy
def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    #SkLearn library imeplementation
    #scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    #Xtrain = scaler.fit_transform(Xtrain)
    
    clf = MultinomialNB(alpha=1.0)
    clf.fit(Xtrain, ytrain)
    
    #predictions = clf.predict(Xtest)
    #print("Predicted: ", predictions)
    #print("Actual: ", ytest)
    
    
    score = clf.score(Xtest, ytest)
    Accuracy = score * 100
    
    return Accuracy

#This function converts a integer matrix to binary matrix
#Input: Integer matrix X
#Output: Binary matrix X
def convert_int_binary(X):
    row_idx = 0
    col_idx = 0

    for row in X:
        col_idx = 0
        for elem in row:
            if elem > 0:
                X[row_idx][col_idx] = 1
            col_idx += 1
        
        row_idx += 1
        
    return X

#Function for training the model - BNBC
#Input: Xtrain, ytrain
#Output: thetaPosTrue, ThetaNegTrue
def naiveBayesBernFeature_train(Xtrain, ytrain):
    
    #Convert the integer count matrix to binary 1/0 matrix
    convert_int_binary(Xtrain)
    #print("Binary: ", Xtrain)
    
    a = 1 #smoothing factor
    
    doc_num = 0
    pos_Xtrain = []
    neg_Xtrain = []
    
    #divide positive and negative cases, this may not be needed if we always assume
    #that the data will be equally distributed and sorted. But, I am generalizing it here
    #in case we get data mixed in
    doc_num = 0
    for docj in Xtrain:
        if ytrain[doc_num] == 1:
            pos_Xtrain.append(docj)
        elif ytrain[doc_num] == -1:
            neg_Xtrain.append(docj)  
        doc_num += 1
    #no.of docs in pos and neg class
    docs_p = len(pos_Xtrain)
    docs_n = len(neg_Xtrain)
    
    #print("Num docs: ", docs_p, docs_n)
    
    #Count docs in class with occurance of word wi 
    pos_doc_cnt = np.sum(pos_Xtrain, axis=0)
    neg_doc_cnt = np.sum(neg_Xtrain, axis=0)
    
    thetaPosTrue = np.divide(np.add(pos_doc_cnt, a), docs_p + 2) 
    thetaNegTrue = np.divide(np.add(neg_doc_cnt, a), docs_n + 2)
    
    return thetaPosTrue, thetaNegTrue
    

def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    
    convert_int_binary(Xtest)
    
    #assuming we will always have 2 classes and equal number of samples in each class
    prior = np.log(0.5)
    
    for docj in Xtest:
        sum_p = prior
        sum_n = prior
        
        # since we have taken biray values of the docj, this operation is same 
        # as sum_p += np.log(thetaPosTrue) for each element in docj
        prod_p = np.multiply(docj, np.log(thetaPosTrue))
        sum_p = np.sum(prod_p)
        
        prod_n = np.multiply(docj, np.log(thetaNegTrue))
        sum_n = np.sum(prod_n)
        
        if sum_p > sum_n:
            yPredict.append(1)
        else:
            yPredict.append(-1)
    
    #print("Predicted Labels:" , yPredict)
    #print("Actual Labels:" , ytest)
    
    total = 0
    for i, j in zip(yPredict, ytest):
        if i == j:
            total +=1
        else:
            pass   
    Accuracy = total/len(ytest) * 100
    
    return yPredict, Accuracy

#Function for training and testing using sklearn
#Input: Xtrain, ytrain, Xtest, ytest
#Output: Accuracy
def naiveBayesMulFeature_sk_BNBC(Xtrain, ytrain, Xtest, ytest):
    #SkLearn library imeplementation

    clf = BernoulliNB(alpha=1.0)
    clf.fit(Xtrain, ytrain)
    
    score = clf.score(Xtest, ytest)
    Accuracy = score * 100
    
    return Accuracy

if __name__ == "__main__":
    '''
    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    '''
    
    textDataSetsDirectoryFullPath = '/Users/vanareddy/Fall2019-ML/ML-PA5/data_sets'
    #textDataSetsDirectoryFullPath = '/Users/vanareddy/Fall2019-ML/ML-PA5/test-data_sets-5'
    #textDataSetsDirectoryFullPath = '/Users/vanareddy/Fall2019-ML/ML-PA5/test-data_sets'

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)
    
    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")
    
    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)
    
    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)
    
    
    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")
    
    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    
    #For verification
    '''
    Accuracy_sk = naiveBayesMulFeature_sk_BNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn BernoulliNB accuracy =", Accuracy_sk)
    '''