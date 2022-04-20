import praw
import json
import pandas as pd
import numpy as np
from datetime import datetime
from submission_list import generate_submission_list
#from praw_scrape1 import create_scrape_dataframes
from praw_scrape7 import scrape_praw_to_db
import hidden
import psycopg2 

print ('Functions loaded')

## Auth codes for API
def lambda_handler(event, context):
    
    
    ### Starting connections and variables needed for our functions and batch logic

    ### Code to connect to Postgres
    pg_secrets = hidden.pg_secrets()
    conn = psycopg2.connect(
        host =      pg_secrets['host'] ,
        port =      pg_secrets['port'],
        database =  pg_secrets['database'], 
        user =      pg_secrets['user'], 
        password =  pg_secrets['password'] , 
        connect_timeout=3)
    
    
    ### Code to connect to Reddit
    r_secrets = hidden.reddit_secrets()
    reddit = praw.Reddit(
        client_id       = r_secrets['APP_ID'],
        client_secret   = r_secrets['APP_SECRET'],
        user_agent      = r_secrets['APP_NAME'],
        username        = r_secrets['REDDIT_USERNAME'],
        password        = r_secrets['REDDIT_PASSWORD'],
        check_for_async = False # This additional parameter supresses some annoying warnings about "asynchronous PRAW " https://asyncpraw.readthedocs.io/en/stable/
    )
    
    ### Set up variables
    # OLD - changed on 11 March 2022 subreddit_scrape_list = ["science","investing", "AskReddit", "news"]
    subreddit_scrape_list = ["investing", "wallstreetbets", "StockMarket", "stocks", "news"]
    n_posts = 1
    n_comments = 5 
    hrs_to_track = 24 #number of hours to track a post/submission
    char_limit = 256 #character limit for the text fields in the database
    time_of_batch = datetime.utcnow().replace(microsecond=0)
    
    print("Submission List Creation START")
    #This tells us what submission/post id's both old & new to get. (there are duplicates that are eliminated next)
    new_submission_list, old_submission_list = generate_submission_list( conn, reddit, time_of_batch, 
                                                                hrs_to_track, n_posts, 
                                                                subreddit_scrape_list) 
                                                                
    print("Submission List Creation END.  # of Submissions that will be attempted:", len( list(set(new_submission_list + old_submission_list))) )
    
    ### re-set the time of batch before our upload
    time_of_batch = datetime.utcnow().replace(microsecond=0)
    print('time_of_batch:', time_of_batch)
    
    ## create batch entry in table
    cur = conn.cursor()  # open the DB connection
    sql = '''
        INSERT into batch(time_of_batch, posts_updated_in_batch, posts_new_in_batch, comments_in_batch)
        VALUES (%s, %s, %s, %s)
        '''
    cur.execute(sql, (time_of_batch, 0, 0, 0))  
    conn.commit()
    cur.close()
    print("Batch Table id created, counts set to zero")
    
    ## run the scrape/load function
    print('Scrape/Upload script Start')
    posts_updated_in_batch, posts_new_in_batch, comments_in_batch = scrape_praw_to_db(conn, reddit, time_of_batch, hrs_to_track, n_comments, char_limit, new_submission_list, old_submission_list )
    print('Scrape/Upload script END.' ,'posts_updated_in_batch', posts_updated_in_batch, 'posts_new_in_batch', posts_new_in_batch, 'comments_in_batch', comments_in_batch)
    
          
    ## update batch entry in table
    cur = conn.cursor()  # open the DB connection
    sql = '''
        UPDATE batch 
        SET posts_updated_in_batch = %s, 
            posts_new_in_batch = %s, 
            comments_in_batch = %s
        WHERE time_of_batch = %s ;
        '''
    cur.execute(sql, (posts_updated_in_batch, posts_new_in_batch, comments_in_batch, time_of_batch))  
    conn.commit()
    cur.close()
    
    print("Batch Table id updated with counts")
    
    
    return "Completed test"
