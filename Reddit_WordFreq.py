# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:31:06 2019

@author: mjh250
"""

import praw
from praw.models import MoreComments
import string
from matplotlib import pyplot as plt
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from rake_nltk import Rake
import re


def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))


def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux


def flattenList(inputList):
    flat_list = [item for sublist in inputList for item in sublist]
    return flat_list


def get_word_list_from_file(filename):
    # with context manager assures us the
    # file will be closed when leaving the scope
    with open(filename) as f:
        for line in f.read().splitlines():
            # yield the result as a generator
            yield line
            
            
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def clean_text(text):
    text = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*(?<![.,])', ' ', text.lower())
    words = re.findall(r'[a-z.,]+', text)
    return ' '.join(words)


def get_ordered_words_from_comment_list(comment_list, printset = string.printable):
    wordlist = flattenList([comment.split() for comment in comment_list])
    for word in wordlist:
        wordset = set(word)
        if not wordset.issubset(printset):
            wordlist.remove(word)
        if wordset.issubset(string.punctuation):
            wordlist.remove(word)  # If the word is just punctuation, remove it
    return wordlist


def remove_stopwords_from_wordlist(wordlist, stopwords=None):
    if stopwords is None:
        stopwords = get_word_list_from_file("Stopwords.txt")
    for word in stopwords:
        while word in wordlist:
            wordlist.remove(word)
    return wordlist


def get_rake_keyphrases_from_text(text, stopwords=None, printset = string.printable):
    if stopwords is None: 
        stopwords = get_word_list_from_file("Stopwords.txt")
    rake_object = Rake(stopwords = stopwords)
    rake_object.extract_keywords_from_text(text)
    rake_keywords = rake_object.get_ranked_phrases_with_scores()
    return rake_keywords


if __name__ == '__main__':
    reddit = praw.Reddit('bot1')
    subreddit = reddit.subreddit("science")
    number_of_submissions = 1
    number_of_comments = 100
    number_of_words = 10
    nltk.data.path.append('./nltk_data/')
    wordnet_lemmatizer = WordNetLemmatizer()
    
    comment_list = []
    for submission in subreddit.hot(limit=number_of_submissions):
        print("Analysing submission with title: ", submission.title)
        top_level_comments = list(submission.comments[0:number_of_comments])
        for comment in top_level_comments:
            if isinstance(comment, MoreComments):
                continue
            comment_list.append(comment.body.lower())
    
    wordlist = get_ordered_words_from_comment_list(comment_list, printset = string.printable)
    
    text = ''
    for comment in comment_list:
        text += comment + ' '
    text = clean_text(text)
    
    rake_keywords = get_rake_keyphrases_from_text(text)
    succint_wordlist = remove_stopwords_from_wordlist(wordlist)
    
    lemmatized_wordlist = []
    for word in wordlist:
        lemmatized_wordlist.append(wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(word)))
            
    lemmatized_dictionary = wordListToFreqDict(lemmatized_wordlist)
    lemmatized_sorteddict = sortFreqDict(lemmatized_dictionary)
    dictionary=wordListToFreqDict(wordlist)
    sorteddict = sortFreqDict(dictionary)
    
    print("Most used words:")
    for word in sorteddict[0:number_of_words]:
        print(word[1].encode('ascii', 'ignore')+" ("+str(word[0])+")")
        
    print("Most used words (lemmatized):")
    for word in lemmatized_sorteddict[0:number_of_words]:
        print(word[1].encode('ascii', 'ignore')+" ("+str(word[0])+")")
     
    top_word_list = [word[1] for word in sorteddict[0:number_of_words]]
    frequency = [word[0] for word in sorteddict[0:number_of_words]]
    indices = np.flip(np.arange(len(top_word_list)), 0)
    plt.barh(indices, frequency, color='r')
    plt.yticks(indices, top_word_list, rotation='horizontal')
    plt.title("Analysis of the frequency of the most used words.")
    plt.xlabel("Number of occurences")
    plt.tight_layout()
    plt.show()
            
    print("Rake keywords:")
    for keyword in rake_keywords[0:number_of_words]:
        print(keyword[1].encode('ascii', 'ignore')+" ("+str(keyword[0])+")")