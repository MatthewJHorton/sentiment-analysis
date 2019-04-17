# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:31:06 2019

@author: mjh250
"""

import praw
from praw.models import MoreComments


if __name__ == '__main__':
    reddit = praw.Reddit('bot1')
    subreddit = reddit.subreddit("spacex")
    
    for submission in subreddit.hot(limit=1):
        print("Title: ", submission.title)
        print("---------------------------------")
        top_level_comments = list(submission.comments[0:4])
        for comment in top_level_comments:
            if isinstance(comment, MoreComments):
                continue
            print(comment.body)
            print("\n")
            
            