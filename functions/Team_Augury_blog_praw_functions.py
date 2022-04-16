import praw
import pandas as pd
import numpy as np  
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from profanity_filter import ProfanityFilter
from functions import Team_Augury_feature_functions
import spacy  #needed for language profanity filtering

### this function creates list of post/submission id's we want

def blog_submission_list(reddit, time_of_batch, hrs_to_track, n_posts, subreddit_scrape_list): 

    ### Generate list of NEW submissions from the subreddits
    new_submission_list = []
    

    for subreddit_scrape_name in subreddit_scrape_list:
        subreddit = reddit.subreddit(subreddit_scrape_name)
        ### POSTS/SUBMISSIONS
        ### using the .hot method pulls the submissions in the same order you would see on Reddit # other options are [controversial, gilded, hot, new, rising, top]
        ### we will scrape "new"
        submission_counter = 0  #initiate a counter
        for submission in subreddit.new(limit=n_posts):
            p_id = submission.id #submission.name #Fullname of the submission.
            p_author = submission.author
            p_created_at = datetime.fromtimestamp(submission.created_utc),  #Time the submission was created, represented in Unix Time
            p_since_created =  int(pd.Series(  (pd.Series(time_of_batch) - pd.Series(p_created_at))  ).astype('timedelta64[h]'))  # creates integer number of hours that have passed using datetime timedelta functionality
            p_current_flag = bool(1 if p_since_created < hrs_to_track else 0)  # is the post current in our system or not

            if p_author==None:  #if an author == None type, that usually means the comment has been deleted by the moderator!
                continue
            elif p_author=="AutoModerator":  # we don't want any auto-posts by the moderator!
                continue
            elif p_current_flag==False:
                continue
            else:
                new_submission_list.append(p_id)
                #print(submission.title)

    return new_submission_list


### this function creates the information from PRAW for both submissions/posts and for comments, in dataframe form.

def blog_scrape_dataframes(reddit, time_of_batch, n_comments, new_submission_list): #, old_submission_list=old_submission_list ):
    submission_df = pd.DataFrame()
    comment_df = pd.DataFrame()

    ### https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html for what items can be pulled for subreddit
    ### https://praw.readthedocs.io/en/stable/code_overview/models/submission.html for what items can be pulled for 'submission' in subreddit
    ### https://praw.readthedocs.io/en/stable/code_overview/models/comment.html for what items can be pulled about the comment/reply 
    ### https://praw.readthedocs.io/en/stable/code_overview/models/redditor.html for what items can be pulled about the redditor (i.e. submission.author or comment.author)

    for i, item in enumerate(new_submission_list):
        p_id = str(item)
        submission = reddit.submission(id=p_id)#create submission instance
        #print(submission.author, '//', submission.title)
        
        if submission.author==None:  #if an author == None type, that usually means the comment has been deleted by the moderator!
            continue
        elif submission.author=="AutoModerator":  # we don't want any auto-posts by the moderator!
            continue
        else: 
            try:
                df = pd.DataFrame(data={
                    'batch_id'          : 1,
                    'time_of_batch'     : time_of_batch,
                    'post_id'              : p_id,   #str(submission.name),  #Fullname of the submission. (different from id, has "t3_" at the start)
                    'hours_since_created'   : 0,
                    #'sr_id'           : str(submission.subreddit.id), # subreddit id
                    'sr'                : str(submission.subreddit.display_name), # subreddit name
                    'post_text'          : str(submission.title), #[:char_limit],  #The title of the submission. #limit the text to the character limit in the database
                    
                    'post_upvotes'         : int(submission.score),  #The number of upvotes for the submission.
                    'number_comments'       : int(submission.num_comments),  #The number of comments on the submission.
                    'post_created_at'      : datetime.fromtimestamp(submission.created_utc),  #Time the submission was created, represented in Unix Time
                    
                    #'p_ratio'           : float(submission.upvote_ratio),  #The percentage of upvotes from all votes on the submission.
                    
                    #'p_redditor_name'   : str(submission.author),  #username
                    #'p_redditor_id'     : str(submission.author.id),  #The ID of the Redditor.
                    'post_author_karma'  : int(submission.author.comment_karma),  #The comment karma for the Redditor.

                }, index=[0])  
                #display(df)
                submission_df = pd.concat([submission_df,df], ignore_index=True)             


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
                            df = pd.DataFrame(data={
                                'batch_id'          : 1,
                                'time_of_batch'     : time_of_batch,
                                'post_id'            : p_id, #the submission id
                                #'sr_id'             : str(submission.subreddit.id), # subreddit id
                                'sr'                : str(submission.subreddit.display_name), # subreddit name

                                
                                'comment_id'              : str(comment.id),   #The ID of the comment.
                                
                                'comment_text'            : str(comment.body),#[:char_limit],  #The body of the comment, as Markdown. #limit the text to the character limit in the database
                                'comment_upvotes'         : int(comment.score),  #The number of upvotes for the comment
                                'comment_created_at'      : datetime.fromtimestamp(comment.created_utc),  #Time the submission was created, represented in Unix Time
                                #'c_redditor_name'   : str(comment.author),  #username
                                #'c_redditor_id'     : str(comment.author.id),  #The ID of the Redditor.
                                'comment_author_karma'  : int(comment.author.comment_karma),  #The comment karma for the Redditor.
                                #'c_is_a'            : bool(comment.is_submitter),  #Whether or not the comment author is also the author of the submission.
                                'hours_since_created'     : 0,


                            }, index=[0])
                            comment_df = pd.concat([comment_df,df], ignore_index=True)

                            comment_counter += 1 # increment the counter
                        except:
                            #print("FAILED to update post -", p_id, " within the comments loop.")
                            continue # we can continue because of our max_comment_counts 


            except:
                #print("FAILED to update post -", p_id)
                continue # we can continue to loop through the list of submission id's even if an update fails.  The failing post will eventually expire in our DB.

   
    return submission_df, comment_df


### Featurization comes from stored functions imported from .py files above.  Comment out any part we don't want.

def blog_feature_creation(post_data, comments_data):
    ### Start by cleaning profanity from posts and comments
    #post_data = Team_Augury_feature_functions.post_profanity_removal(post_data.copy())
    #display(post_data)
    #comments_data = Team_Augury_feature_functions.comment_profanity_removal(comments_data.copy())
    #display(comments_data)

    ### Basic feature addtions for post-only data
    feature_df = Team_Augury_feature_functions.post_basic_features(post_data.copy())
    ### Basic feature additions for comments (avg's)
    feature_df = feature_df.merge( Team_Augury_feature_functions.comment_basic_features(comments_data.copy()) , how='left',on=['batch_id','post_id'])
    ### VADER sentiment for posts
    feature_df = feature_df.merge( Team_Augury_feature_functions.post_sentiment_func(post_data.copy()) , how='left',on=['post_id', 'post_text'])
    ### VADER sentiment for comments (avg)
    feature_df = feature_df.merge( Team_Augury_feature_functions.comment_sentiment_func(comments_data.copy()) , how='left',on=['batch_id','post_id'])
    ### SBERT sentence transform for posts
    feature_df = feature_df.merge( Team_Augury_feature_functions.post_sentence_transform_func(post_data.copy()) , how='left',on=['post_id'])
    ### SBERT sentence transform for comments (avg's)
    feature_df = feature_df.merge( Team_Augury_feature_functions.comment_sentence_transform_func(comments_data.copy()) , how='left',on=['batch_id','post_id'])

    ### check for NaN's
    #display(feature_df[feature_df.isna().any(axis=1)])
    feature_df.replace([np.nan], 0, inplace=True) #clean NaN's formed from comments merging
    return feature_df

