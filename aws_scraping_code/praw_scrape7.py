import praw
import json
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2 



## MODULE
### this function creates the information from PRAW for both submissions/posts and for comments AND puts directly into the database

def scrape_praw_to_db(conn, reddit, time_of_batch, hrs_to_track, n_comments, char_limit, new_submission_list, old_submission_list ):

    ### start by combining the 'old' and 'new' submission id list.  We keep the lists separate for de-bugging purposes.
    submission_list = list(set(new_submission_list + old_submission_list))
    #print(submission_list)

    #submission_df = pd.DataFrame()
    #comment_df = pd.DataFrame()

    comments_in_batch = 0
    posts_new_in_batch = 0
    posts_updated_in_batch = 0

    cur = conn.cursor() # open the DB connection

    ### https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html for what items can be pulled for subreddit
    ### https://praw.readthedocs.io/en/stable/code_overview/models/submission.html for what items can be pulled for 'submission' in subreddit
    ### https://praw.readthedocs.io/en/stable/code_overview/models/comment.html for what items can be pulled about the comment/reply 
    ### https://praw.readthedocs.io/en/stable/code_overview/models/redditor.html for what items can be pulled about the redditor (i.e. submission.author or comment.author)

    for i, item in enumerate(submission_list):
        p_id = str(item)
        submission = reddit.submission(id=p_id[3:]) #create submission instance, removing the 't3_' from the id


        if submission.author==None:  #if an author == None type, that usually means the comment has been deleted by the moderator!
            continue
        elif submission.author=="AutoModerator":  # we don't want any auto-posts by the moderator!
            continue
        else: 
            try:
                # submission_df = submission_df.append({

                #     'time_of_batch'     : time_of_batch,
                    
                #     'sr_id'           : str(submission.subreddit.id), # subreddit id
                #     'sr_name'         : str(submission.subreddit.display_name), # subreddit name

                #     'p_created_at'      : datetime.fromtimestamp(submission.created_utc),  #Time the submission was created, represented in Unix Time
                #     'p_id'              : p_id,   #str(submission.name),  #Fullname of the submission. (different from id, has "t3_" at the start)
                #     'p_number_comments' : int(submission.num_comments),  #The number of comments on the submission.
                #     'p_upvotes'         : int(submission.score),  #The number of upvotes for the submission.
                #     'p_ratio'           : float(submission.upvote_ratio),  #The percentage of upvotes from all votes on the submission.
                #     'p_text'            : str(submission.title)[:char_limit],  #The title of the submission. #limit the text to the character limit in the database
                #     'p_redditor_name'   : str(submission.author),  #username
                #     'p_redditor_id'     : str(submission.author.id),  #The ID of the Redditor.
                #     'p_redditor_karma'  : int(submission.author.comment_karma),  #The comment karma for the Redditor.

                #     'p_since_created'   : int(pd.Series(  (pd.Series(time_of_batch) - pd.Series(datetime.fromtimestamp(submission.created_utc)))  ).astype('timedelta64[h]')), # creates integer number of hours that have passed using datetime timedelta functionality
                #     'p_current_flag'    : True ,
                #     'p_old_or_new'      : 'OLD' if p_id in old_submission_list else 'NEW',

                # }, ignore_index=True)
                
                #print("Start REDDIT submission data download")
                
                ### submission variables
                time_of_batch     = time_of_batch
                
                sr_id             = str(submission.subreddit.id) # subreddit id
                sr_name           = str(submission.subreddit.display_name) # subreddit name
                #print('ok - sr info')

                p_created_at      = datetime.fromtimestamp(submission.created_utc)  #Time the submission was created, represented in Unix Time
                #print('ok - p_created_at')
                p_id              = p_id   #str(submission.name),  #Fullname of the submission. (different from id, has "t3_" at the start)
                p_number_comments = int(submission.num_comments)  #The number of comments on the submission.
                p_upvotes         = int(submission.score)  #The number of upvotes for the submission.
                p_ratio           = float(submission.upvote_ratio)  #The percentage of upvotes from all votes on the submission.
                p_text            = str(submission.title)[:char_limit]  #The title of the submission. #limit the text to the character limit in the database
                p_redditor_name   = str(submission.author)  #username
                p_redditor_id     = str(submission.author.id)  #The ID of the Redditor.
                p_redditor_karma  = int(submission.author.comment_karma)  #The comment karma for the Redditor.
                #print('ok - p_redditor_karma')
                p_since_created   = int(pd.Series(  (pd.Series(time_of_batch) - pd.Series(p_created_at))  ).astype('timedelta64[h]'))# creates integer number of hours that have passed using datetime timedelta functionality
                #print('ok - p_since_created')
                p_current_flag    = True 
                #print('ok - p_current_flag')
                p_old_or_new      = 'OLD' if p_id in old_submission_list else 'NEW' 
                
                
                #print("END REDDIT submission data download")
                #print("Start SQL submission data upload")

                #print('Submission loop - table:','subreddit_info')
                sql = '''
                    INSERT into subreddit_info(sr_id, sr_name)
                    VALUES (%s, %s)
                    ON CONFLICT (sr_id)
                    DO NOTHING;
                '''
                cur.execute(sql, (sr_id, sr_name))
                conn.commit()
                #print('Submission loop - table:','redditor')
                sql = '''
                    INSERT into redditor(author_id, author_name)
                    VALUES (%s, %s)
                    ON CONFLICT(author_id)
                    DO NOTHING; 
                '''                  
                cur.execute(sql, (p_redditor_id, p_redditor_name))
                conn.commit()
                #print('Submission loop - table:','post')
                sql = '''
                        INSERT into post(post_id, sr, created_at, author_id, post_text, is_current)
                        VALUES (%s,(SELECT id from subreddit_info WHERE sr_id = %s), %s ,(SELECT id from redditor WHERE author_id = %s),%s,%s)
                        ON CONFLICT (post_id)
                        DO NOTHING;    
                    '''                   
                cur.execute(sql, (p_id,
                                    sr_id,
                                    p_created_at,
                                    p_redditor_id, 
                                    p_text,
                                    True #bool(p_current_flag)
                                    )) 
                conn.commit()
                #print('Submission loop - table:','post_over_time')
                sql = '''
                        INSERT into post_over_time(post_id, batch_id, upvotes, ratio, hours_since_created, number_comments)
                        VALUES ((SELECT id from post WHERE post_id = %s),(SELECT id from batch WHERE time_of_batch = %s), %s, %s,%s,%s)
                    '''
                cur.execute(sql, (p_id,
                                    time_of_batch, 
                                    int(p_upvotes),
                                    float(p_ratio), 
                                    int(p_since_created),
                                    int(p_number_comments), 
                                    )) 
                conn.commit()
                #print('Submission loop - table:','redditor_over_time')
                sql = '''
                        INSERT into redditor_over_time(author_id, batch_id, karma)
                        VALUES ((SELECT id from redditor WHERE author_id = %s),(SELECT id from batch WHERE time_of_batch = %s), %s); 
                    '''
                cur.execute(sql, (p_redditor_id,
                                time_of_batch, 
                                float(p_redditor_karma)
                                ))
                conn.commit()

                
                
                if p_old_or_new == 'OLD':
                    posts_updated_in_batch += 1
                else:
                    posts_new_in_batch += 1




                ### COMMENTS under the posts.  We scrape the "top-level" comments.
                submission.comment_sort = "top" # Set a sort method, default is 'confidence' # sorting options are [confidence, top, new, controversial, old, random, qa, live, blank]
                submission.comments.replace_more(limit=0) # this line removes any MoreComment instances (along the lines of a "View More Comments" button at the bottom of a webpage)

                comment_counter = 0  #initiate a counter
                max_comment_counts = n_comments + 3  #to prevent an infinite loop we allow for "3 strikes", or 3 extra loops to account for an Automoderator or deleted comment. 
                #for comment in submission.comments[0:n_comments]:
                for j, comment in enumerate(submission.comments):
                    
                    if comment.author==None:  #if an author == None type, that usually means the comment has been deleted by the moderator!
                        continue
                    elif comment.author=="AutoModerator":  # we don't want any auto-comments by the moderator!
                        continue
                    elif (comment_counter < n_comments) & (j < max_comment_counts):
                        try:
                            # comment_df = comment_df.append({

                            #     'time_of_batch'     : time_of_batch,
                                
                            #     'sr_id'             : str(submission.subreddit.id), # subreddit id
                            #     'sr_name'           : str(submission.subreddit.display_name), # subreddit name

                            #     'p_id'              : p_id, #the submission id

                            #     'c_created_at'      : datetime.fromtimestamp(comment.created_utc),  #Time the submission was created, represented in Unix Time
                            #     'c_id'              : str(comment.id),   #The ID of the comment.
                            #     'c_upvotes'         : int(comment.score),  #The number of upvotes for the comment
                            #     'c_text'            : str(comment.body)[:char_limit],  #The body of the comment, as Markdown. #limit the text to the character limit in the database
                            #     'c_redditor_name'   : str(comment.author),  #username
                            #     'c_redditor_id'     : str(comment.author.id),  #The ID of the Redditor.
                            #     'c_redditor_karma'  : int(comment.author.comment_karma),  #The comment karma for the Redditor.
                            #     'c_is_a'            : bool(comment.is_submitter),  #Whether or not the comment author is also the author of the submission.
                            #     'c_since_created'   : int(pd.Series(  (pd.Series(time_of_batch) - pd.Series(datetime.fromtimestamp(comment.created_utc)))  ).astype('timedelta64[h]')), # creates integer number of hours that have passed using datetime timedelta functionality

                            # }, ignore_index=True)

                            time_of_batch     = time_of_batch
                            
                            #sr_id             = str(submission.subreddit.id) # subreddit id
                            #sr_name           = str(submission.subreddit.display_name) # subreddit name

                            p_id              = p_id #the submission id

                            c_created_at      = datetime.fromtimestamp(comment.created_utc)  #Time the submission was created, represented in Unix Time
                            c_id              = str(comment.id)   #The ID of the comment.
                            c_upvotes         = int(comment.score)  #The number of upvotes for the comment
                            c_text            = str(comment.body)[:char_limit]  #The body of the comment, as Markdown. #limit the text to the character limit in the database
                            c_redditor_name   = str(comment.author)  #username
                            c_redditor_id     = str(comment.author.id)  #The ID of the Redditor.
                            c_redditor_karma  = int(comment.author.comment_karma)  #The comment karma for the Redditor.
                            c_is_a            = bool(comment.is_submitter)  #Whether or not the comment author is also the author of the submission.
                            c_since_created   = int(pd.Series(  (pd.Series(time_of_batch) - pd.Series(datetime.fromtimestamp(comment.created_utc)))  ).astype('timedelta64[h]')) # creates integer number of hours that have passed using datetime timedelta functionality
                        

                            #print('Comment loop - table:','redditor')
                            sql = '''
                                INSERT into redditor(author_id, author_name)
                                VALUES (%s, %s)
                                ON CONFLICT(author_id)
                                DO NOTHING; 
                            '''                  
                            cur.execute(sql, (c_redditor_id, c_redditor_name))
                            conn.commit()
                            #print('Comment loop - table:','comments')
                            sql = '''
                                INSERT into comments(comment_id, post_id, created_at, author_id, comment_text, commenter_is_author)
                                VALUES (%s,(SELECT id from post WHERE post_id = %s),%s, (SELECT id from redditor WHERE author_id = %s),%s,%s)
                                ON CONFLICT (comment_id)
                                DO NOTHING;    
                            '''     
                            cur.execute(sql, (c_id,
                                        p_id,
                                        c_created_at,
                                        c_redditor_id, 
                                        c_text,
                                        bool(c_is_a)
                                        )) 
                            conn.commit()
                            #print('Comment loop - table:','comments_over_time')
                            sql = '''
                                INSERT into comments_over_time(comment_id, batch_id, upvotes, hours_since_created)
                                VALUES ((SELECT id from comments WHERE comment_id = %s),(SELECT id from batch WHERE time_of_batch = %s), %s, %s)
                            '''
                            cur.execute(sql, (c_id,
                                            time_of_batch, 
                                            int(c_upvotes), 
                                            int(c_since_created)
                                            )) 
                            conn.commit()
                            #print('Comment loop - table:','redditor_over_time')
                            sql = '''
                                    INSERT into redditor_over_time(author_id, batch_id, karma)
                                    VALUES ((SELECT id from redditor WHERE author_id = %s),(SELECT id from batch WHERE time_of_batch = %s), %s)
                                '''
                            cur.execute(sql, (c_redditor_id,
                                            time_of_batch, 
                                            float(c_redditor_karma)
                                            ))
                            conn.commit() 

                            comment_counter += 1 # increment the counter
                            comments_in_batch += 1
                            
                            print('Submission index:', i, 'post:', p_id, 'posts_updated:', posts_updated_in_batch, 'posts_new:', posts_new_in_batch, 'Comment index:', j,'comments_in_batch:', comments_in_batch)
                            

                        except:
                            print("FAILED to update post -", p_id, " within the comments loop.", 'Submission index:', i, 'Comment index:', j,'comments_in_batch:', comments_in_batch)
                            continue # we can continue because of our max_comment_counts 


            except:
                print("FAILED to update post -", p_id)
                continue # we can continue to loop through the list of submission id's even if an update fails.  The failing post will eventually expire in our DB.



    ## ### add some datetime based columns and any additional information for debugging
    ## submission_df['p_since_created'] = (submission_df['time_of_batch']-submission_df['p_created_at']).astype('timedelta64[h]').astype('int')
    ## submission_df['p_current_flag'] = submission_df.apply(lambda x: True if x['p_since_created'] <= hrs_to_track else False, axis=1).astype('bool')
    ## submission_df['p_old_or_new'] = submission_df.apply(lambda x: 'OLD' if x['p_id'] in old_submission_list else 'NEW', axis=1).astype('str')
    
    ## comment_df['c_since_created'] = (comment_df['time_of_batch']-comment_df['c_created_at']).astype('timedelta64[h]').astype('int')

    #display(submission_df)
    #display(comment_df)

    ### close the connection
    cur.close()
    return posts_updated_in_batch, posts_new_in_batch, comments_in_batch

