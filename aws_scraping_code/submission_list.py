### MODULE
### this function creates list of post/submission id's we want, both NEW and OLD, to send to our database

import praw
import json
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2 

def generate_submission_list( conn, reddit, time_of_batch, hrs_to_track, n_posts, subreddit_scrape_list):
    

    ### Generate list of NEW submissions from the subreddits
    new_submission_list = []
    old_submission_list = []

    for subreddit_scrape_name in subreddit_scrape_list:
        subreddit = reddit.subreddit(subreddit_scrape_name)
        ### POSTS/SUBMISSIONS
        ### using the .hot method pulls the submissions in the same order you would see on Reddit # other options are [controversial, gilded, hot, new, rising, top]
        ### we will scrape "new"
        submission_counter = 0  #initiate a counter
        for submission in subreddit.new():

            if submission.author==None:  #if an author == None type, that usually means the comment has been deleted by the moderator!
                continue
            elif submission.author=="AutoModerator":  # we don't want any auto-posts by the moderator!
                continue
            elif submission_counter < n_posts:
                p_id = submission.name #Fullname of the submission.
                new_submission_list.append(p_id)
                submission_counter += 1

    ### Generate list of OLD submissions that are still CURRENT to be updated
    sql = '''
        SELECT post_id, created_at from post WHERE is_current = True ;   
        '''
    cur = conn.cursor()
    cur.execute(sql)
    submission_update_items = cur.fetchall()

    ## find expired submissions, set to expired
    for item in submission_update_items:
        #print( item[1], time_of_batch)
        p_id = str(item[0])
        p_created_at = item[1]
        p_since_created =  int(pd.Series(  (pd.Series(time_of_batch) - pd.Series(p_created_at))  ).astype('timedelta64[h]'))  # creates integer number of hours that have passed using datetime timedelta functionality
        p_current_flag = bool(1 if p_since_created <= hrs_to_track else 0)  # is the post current in our system or not
        #print(p_id, time_of_batch, p_created_at, p_since_created, p_current_flag)
        
        if p_current_flag == False:
            sql = '''
                UPDATE post SET is_current = False WHERE post_id = %s ;   
            '''
            cur.execute(sql, (p_id, ))
            conn.commit()

        elif p_current_flag == True:
            submission = reddit.submission(id=p_id[3:]) #create submission instance, removing the 't3_' from the id

            if submission.author==None:  #if an author == None type, that usually means the comment has been deleted by the moderator!
                sql = '''
                    UPDATE post SET is_current = False WHERE post_id = %s ;   
                '''
                cur.execute(sql, (p_id, ))
                conn.commit()
                continue
            elif submission.author=="AutoModerator":  # we don't want any auto-posts by the moderator!
                continue
                sql = '''
                    UPDATE post SET is_current = False WHERE post_id = %s ;   
                '''
                cur.execute(sql, (p_id, ))
                conn.commit()
            else:
                old_submission_list.append(p_id)


    cur.close()
    #submission_list = new_submission_list + old_submission_list
    #print(submission_list)
    return new_submission_list, old_submission_list


#generate_submission_list()