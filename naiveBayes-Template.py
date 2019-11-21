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
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB


###############################################################################

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

def similar_text(word, the_word):
   
    similar = 0.0
    
    if word[:len(word)-1] == the_word[:len(word)-1] :
        word = word + ' ' * (len(the_word) - len(word)-1)
        the_word = the_word + ' ' * (len(word) - len(the_word))
        similar = sum(1 if i == j else 0 for i, j in zip(word, the_word)) / float(len(word))
    
    return similar

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

def extract_and_match_text(vocabulary, text):
    
    #split the text into words
    words = text.split()
    #print(words)
    matched_text = match_text(vocabulary, words)
    #print(matched_text)
    
    return matched_text

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
            
        BOWDj[-1] = len(matched_text) - found
        
        #print(np.array(BOWDj))
        
    elif (choice == 2):
        #NLTK
        stop_words = []
        #print("Choice 2")
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
            
        print(len(stemmed_text))
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


def loadData(Path):
    
    #open dictionary and read into standard vocabulary
    dictionary = os.path.join(Path, '../dictionary.txt')
    with open(dictionary, 'r') as file:
            vocab = file.read().split()
    
    #print("Vocabulary:\n", vocab)
    
    Xtrain, Xtest, ytrain, ytest = [], [], [], []
    data = []
    #Process Taining Data
    print("\nGetting Training data")
    training_data = os.path.join(Path, 'training_set')
    for rev_type in os.listdir(training_data):

        if(rev_type == 'pos'):
            y_value = 1
        else:
            y_value = -1
            
        rev_type = os.path.join(training_data, rev_type)
        for docj in os.listdir(rev_type):
            docj = os.path.join(rev_type, docj)
            #Xtrain.append(transfer(docj, vocab, 1))
            #ytrain.append(y_value)
            data.append([transfer(docj, vocab, 1), y_value])
    
    #print("Shuffling xtrain and ytrain together")
    #random.shuffle(data)
    for x, y in data:
        Xtrain.append(x)
        ytrain.append(y)
    
    #print(Xtrain[0])
    #print(ytrain)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    
    print("Xtrain: ", Xtrain.shape)
    print("Ytrain: ", ytrain.shape)
    
    
    data = []
    #Process Test Data
    print("\nGetting Test data")
    test_data = os.path.join(Path, 'test_set')
    for rev_type in os.listdir(test_data):

        if(rev_type == 'pos'):
            y_value = 1
        else:
            y_value = -1
            
        rev_type = os.path.join(test_data, rev_type)
        for docj in os.listdir(rev_type):
            docj = os.path.join(rev_type, docj)
            #Xtest.append(transfer(docj, vocab, 1))
            #ytest.append(y_value)
            data.append([transfer(docj, vocab, 1), y_value])
    
    print("Shuffling xtrain and ytrain together")
    random.shuffle(data)
    for x, y in data:
        Xtest.append(x)
        ytest.append(y)
    
    #print(Xtest)
    #print(ytest)
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    
    #print(Xtest.shape)
    #print(ytest.shape)
    
    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    
    #My implementation
    #Calculate P(Wi/cj), probability of word wi for given class cj
    doc_num = 0
    a = 1 #smoothing factor
    n = Xtrain.shape[1] # no.of words in vocabulary
    d = 0 #Sum of all words in a given class, calculated below
    
    print(n)
    
    print(ytrain)
    doc_num = 0
    pos_Xtrain = []
    neg_Xtrain = []
    #divide positive and negative cases, this may not be needed if we always assume
    #that the data will be equally distributed and sorted. But I am generalizing it here
    for docj in Xtrain:
        if ytrain[doc_num] == -1:
            pos_Xtrain.append(docj)
        else:
            neg_Xtrain.append(docj)  
        doc_num += 1
    
    #add all the counts of wi
    pos_Xtrain = np.sum(pos_Xtrain, axis=0)
    #print(pos_Xtrain)
    d = np.sum(pos_Xtrain)
    n_d = n + d
    #add laplace smoothing
    pos_Xtrain = np.add(pos_Xtrain, a)
    thetaPos = np.divide(pos_Xtrain, n_d)
    #print("----Positive-----")
    #print(thetaPos)
    #add all the counts of wi
    neg_Xtrain = np.sum(neg_Xtrain, axis=0)
    d = np.sum(pos_Xtrain)
    n_d = n + d
    #add laplace smoothing
    neg_Xtrain = np.add(neg_Xtrain, a)
    thetaNeg = np.divide(neg_Xtrain, n_d)
    print("----Negative-----")
    print(thetaNeg)
    
    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    yPredict = []
    Accuracy = 0.0

    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    #Check with library imeplementation

    clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    clf.fit(Xtrain, ytrain)
    print(clf.predict(Xtest))
    print("Actual")
    print(ytest)

    return Accuracy

'''
def naiveBayesBernFeature_train(Xtrain, ytrain):

    #return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    
    #return yPredict, Accuracy
'''

if __name__ == "__main__":
    '''
    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    '''
    
    #textDataSetsDirectoryFullPath = '/Users/vanareddy/Fall2019-ML/ML-PA5/data_sets'
    textDataSetsDirectoryFullPath = '/Users/vanareddy/Fall2019-ML/ML-PA5/test-data_sets'


    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)

    '''
    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")
    
    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)
    '''
    
    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)

    '''
    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")
    
    '''
