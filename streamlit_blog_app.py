import streamlit as st
#from torch import hann_window
#import pandas as pd
#import joblib
#from sklearn.linear_model import LogisticRegression
from PIL import Image
import praw
from profanity_filter import ProfanityFilter

# Title
st.title("Project Augury: Predicting which Investing posts on Reddit are likely to become popular")
st.caption(" **augury** _noun_; a sign of what will happen in the future     -- Cambridge Dictionary")
st.markdown(">Course: SIADS 697/698  \n>> Git Repository: [Github Link](https://github.com/rosecorn24601/Capstone-697-TeamAugury)  \n>> Blog Post: [Streamlit Link](https://share.streamlit.io/rosecorn24601/capstone-697-teamaugury/main/streamlit_blog_app.py)  \n>Authors:  \n>> Antoine Wermenlinger (awerm@umich.edu)  \n>> Chris Lynch (cdlynch@umich.edu)  \n>> Erik Lang (eriklang@umich.edu)")

st.header("Summary")
st.write('''
    Placeholder text
    ''')

st.header("Background")
st.subheader("Motivation")
st.markdown('''
    Reddit is a popular social media networking site. From an investing perspective there are a few subreddits that have discussions related to investing.  A subreddit is a forum on Reddit that is dedicated to a specific topic, in our case we have chosen four subreddits to investigate:

     - r/investing
     - r/wallstreetbets
     - r/StockMarket
     - r/stocks

    WSB achieved some notoriety during the “GameStop” incident where Keith Gill, a formerly little known trader achieved gains of $40M plus in a short period of time referred to as the ‘Reddit Rally’. Reuters claimed his ‘punchy move’ sparked thousands of comments on the WSB subreddit, causing his post(s) to go viral and his stock position to dramatically rise in value.  

    Project Augury is focussed on exploring what makes a post on these subreddits to be popular. We are looking at a group of four subreddits as each subreddit itself has relatively low volumes of posts each day, when compared to the biggest subreddit’s like AskReddit, and our background research confirmed that predictive tasks on social media work better in thematic subreddits and do not necessarily generalize from one theme or subreddit to another.
    ''')



st.subheader("") #create blank space
st.subheader("Related Work")
st.markdown('''
    The table below summarizes the papers we reviewed and the main insights we derived for use in our Augury project.  The types of works span from Supervised Learning methods through to Reinforcement Learning and we learned a little from each.  *(Full Citations are given in an appendix)*
    ''')
related_work = st.container()
with related_work:
    related_work.info('''
        **Title:** Predicting the Popularity of Reddit Posts with AI.[3]  
        **Topic:** Supervised Learning approaches to predict popularity    
        **Implication for our project:** This is a prediction task on similar data.  The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF (Term Frequency-Inverse Document Frequency), and LDA (Latent Dirichlet Allocation) trained on features extracted with Naive Bayes and SVM.  This work is based on regression, trying to predict the number of upvotes, whereas Augury aims to predict whether a post will be popular or not ( a classification problem) within a three hour window.   
        ''')
    related_work.info('''
        **Title:** Data Stories. We analyzed 4 million data points to see what makes it to the front page of reddit. Here’s what we learned.[4]  
        **Topic:** Supervised Learning approaches to predicting Reddit comment Popularity    
        **Implication for our project:** Comparable aims, different NLP and looked only at Comments rather than posts.  Handling similar data, the author aimed to predict comment popularity.  Achieved relatively low accuracy scores ranging from 42% to 52.7% with a Decision Tree Classifier performing best. Cohen's Kappa statistic is applied to show results were, in fact, not much better than random.  In their conclusions they suggest research looks at temporal proximity of comments to posts, a key feature in Augury.
        ''')
    related_work.info('''
        **Title:** Popularity prediction of reddit texts.[5]  
        **Topic:** Supervised learning approach to predict Reddit post popularity    
        **Implication for our project:** Comparable objectives, uses different NLP and features. Focuses on using Topics to determine predictive task.   Achieved 60-75% accuracy on the task, using Latent Dirichlet Allocation (LDA) and Term Frequency Inverse Document Frequency (TFIDF) to classify topics in posts to explore the relationship between topics and posts in order to predict using Naive Bayes and Support Vector Machine Classifiers what will become popular. Augury includes topic modeling as a feature, and our initial model suite included these classifiers.  We later rejected these models as Tree based classifiers proved more performant.
        ''')
    related_work.info('''
        **Title:** Predicting the Popularity of Reddit Posts.[6]  
        **Topic:** Supervised learning approach to predict Reddit post popularity    
        **Implication for our project:** Conducted similar time of day, day of week features to Augury. Also performed sentiment analysis, with a different method. Finally treated the problem as a regression rather than classification one.  Our early experiments found classification to be better suited to our objective.
        ''')
    related_work.info('''
        **Title:** Deepcas: An end-to-end predictor of information cascades.[7]  
        **Topic:** Neural Network approach to predicting information cascades    
        **Implication for our project:** The prediction task in DeepCas was quite different to Augury. The problem definition included a Markov Decision Process as a ‘deep walk path’ making the work potentially relevant when we explored Reinforcement Learning approaches.  Eventually we moved away from these methods as our actor ‘choices’ i.e. picking a post had very little effect on the State/Environment hence we reject RL methods, despite a thorough investigation of use cases and an investigation of relevant works such as those in [8] to [11] below. The RL approach is effectively too contrived for our objective.
        ''')
    related_work.info('''
        **Title:** Deep reinforcement learning with a combinatorial action space for predicting popular reddit threads.[8]  
        **Topic:** Reinforcement Learning on Reddit data    
        **Implication for our project:** Similar domain space, different approaches.  Showed how a simulator might be used to reconstruct ‘Trees’ to set up and test sequential decision making. Related to our main task but not identical.
        ''')
    related_work.info('''
        **Title:** Deep reinforcement learning with a natural language action space.[9]  
        **Topic:** Reinforcement Learning for NLP - Text based games    
        **Implication for our project:** Illustrated the large action space issue for deep Q-learning in NLP.  Helps understand why the Tree approach was taken in [8] in order to re-use approach from text based games when seeking to predict karma on reddit.  
        ''')
    related_work.info('''
        **Title:** Deep reinforcement learning for NLP.[10]  
        **Topic:** A primer on Reinforcement Learning for NLP    
        **Implication for our project:** Simple introduction and overview to papers in the domain.  
        ''')
    related_work.info('''
        **Title:** SocialSift: Target Query Discovery on Online Social Media With Deep Reinforcement Learning.[11]  
        **Topic:** Generation of SQL queries using Reinforcement Learning    
        **Implication for our project:**  Sets out an online method (via API) of testing created text (‘Queries’) where the returned results are classified to create a reward for the RL policy Pi. The text of the query is effectively keywords that are extracted from the corpus of the previous query history and returned results.
        ''')
    related_work.info('''
        **Title:** Real-Time Predicting Bursting Hashtags on Twitter.[12]  
        **Topic:** Predicting hashtag bursts on Reddit    
        **Implication for our project:**  Similar data and has a temporal aspect to the prediction challenge. The definition of a ‘burst’ in this paper used a maximum function of hashtag counts over a 24 hour period to define a burst. Our early exploration of popularity as defined in Augury took inspiration from this, but was later adapted to our 3 hour target which better suited our objective. Augury also took some inspiration from the classification approach used in this paper.
        ''')

st.subheader("") #create blank space
st.subheader("Ethical Considerations")
st.write('''
    There are clearly ethical implications relating from broadcasting messages on social media and the related investments to which these messages refer, as highlighted by the ‘Reddit Rally’ described above. In project Augury we are only looking at the popularity of posts, and we have not correlated this to market activity, which could be an extension of this work.  To some extent this research has already been investigated in XXX   . Therefore we have made no further mitigations in our project related to market ethics.

Another consideration is the free and open nature of online social media, and Reddit in particular. This can, and does, lead to environments which can become toxic in a number of ways. The subreddits which are looking at Investment are typically more professional in nature so the main way in which toxicity occurs in these fora is through the use of profane language. In our pipeline we have removed this language from our analysis using the python module profanity-filter 1.3.3 which replaces profane words with the character “*”.  

    ''')


st.subheader("") #create blank space
st.header("Our Project Workflow")
st.write("The below graphic illustrates our project Augury workflow.  Below we will provide more details on each component of the workflow.")
project_pipeline_image = Image.open('blog_assets/project_pipeline.png')
st.image(project_pipeline_image, caption='Project Augury Workflow')
st.subheader("") #create blank space
st.subheader("Scraping Reddit Data") 


# st.markdown('''
#     The data scrape we finally put into use called upon the PRAW library to make efficient requests to Reddit’s API.  Our approach to scraping was informed by the following article’s XXXX and the code which we eventually used can be found on the project repo, linked at the head of this page. We knew that we wanted to capture the evolution of a posts popularity over time in order to see how and when popular posts developed, so we designed a pipeline that:
#     ''')
# st.markdown('''
#     - Captured a sample of ‘new posts’ each hour.
#         - These were initially five posts per hour, but the consequences of the remainder of our cleaning function meant that this process would ‘timeout’ on AWS Lambda.  So we decided at the end of February 2022 to strip our ‘new posts’ back to one per hour.
#     - Tracked these posts over a 24 hour period.
#         - Each subsequent scrape would recapture the data related to our target posts, such as the number of comments, the text of those comments, the karma (for which you might assume ‘popularity’) of the comment and post authors.
#     - Cleaned the data
#         -  Before we stored the data onto our AWS RDS (a postgresql instance) our code did a lot of the early cleaning work for us.  Initially our requests backed libraries wrote large JSON files to our database, when we went into production we had developed code that extracted and formatted the data we wanted from the subreddit’s, making downstream processing of the data much slicker.  For example XXXX 
#     - Load the data
#         - Our database design/schema of tables is shown below.  This design was to optimize the functioning of the RDS and minimize storage, by reducing duplication to a minimum.


    # ''')
db_schema_image = Image.open('blog_assets/db_schema.png')
st.image(db_schema_image, caption='Augury Database Schema in AWS')


st.markdown('''
    - Extracting and cleaning the data
    - Feature Engineering
        - placeholder
    - modeling
        - placeholder
    - prediction / production
        - placeholder

    ''')

st.subheader("") #create blank space
st.subheader("Feature Engineering (Option 1)")
st.write("After experimentation on our scraped dataset we decided upon the following features:")
feature_table = st.container()
with feature_table:
    with feature_table.expander("Number of comments per hour"):
        st.markdown('''
            *Description:*  This is a count of the comments each post has received, divided by the number of hours that have elapsed since the post was created.  
            *Rationale:*  Our research and intuition told us that the number of people commenting on a post is an indicator of likely popularity.
        ''')
    with feature_table.expander("Author Karma for the Post"):
        st.markdown('''
            *Description:*  We tracked the karma  of both comment and post authors at the time of making either a post or a comment.  
            *Rationale:*  Whilst people who have high Karma scores aren't necessarily ‘influencers’ in the normal social media sense of the word, their karma scores are a good proxy for this.  Our EDA looked to see if posts that were posted by ‘high karma’ authors were more likely to become popular as a result and whilst the correlation was surprisingly low we took this feature forward to the modeling stage to test if this contained any ‘signal’ for our predictive task.
        ''')
    with feature_table.expander("Hour and Day the Post was created"):
        st.markdown('''
            *Description:*  We recorded the hour that a post was made (UTC) to see the correlation with post popularity.  In our pipeline we ‘one hot’ encoded these features before passing them to our training/inference models.  
            *Rationale:*  These features have shown predictive power in other social media analytics tasks [2]. 
        ''')
    with feature_table.expander("VADER Text Sentiment of the Post"):
        st.markdown('''
            *Description:*  We used the VADER sentiment library to classify the sentiment of each posts text.  This produced a value in the range of -1, +1.     
            *Rationale:*  We believe text that has a strongly positive or negative sentiment is more likely to become popular than something that is neutral. 
        ''')
    with feature_table.expander("SBERT Sentence Embeddings of the Post"):
        st.markdown('''
            *Description:*  We used the SBERT library to encode the text of both Posts and Comments.  The SBERT interface was simple to use and produced in effect 380+ features for our classifiers for each post.     
            *Rationale:*  NEEDS UPDATE the rich meaning from language encoded via SBERT, which is based on the state of the art BERT language model.
        ''')
    with feature_table.expander("Average Upvotes for the Top 5 Comments on the Post (per Hour)"):
        st.markdown('''
            *Description:*  We look at the top 5 comments for a post (if available) and see how many upvotes that comment has gotten.     
            *Rationale:*  Posts that gather comments quickly will likely have their popularity influenced by upvotes to those comments. 
        ''')
    with feature_table.expander("Average Author Karma for the Top 5 Comments on the Post"):
        st.markdown('''
            *Description:*  We look at the Commentor Karma for the top 5 comments for a post (if available).     
            *Rationale:*  Posts that gather comments quickly from authors with a high reputation should impact the Post's popularity. 
        ''')
    with feature_table.expander("Average VADER Text Sentiment of the Top 5 Comments of the Post"):
        st.markdown('''
            *Description:*  We look at the average VADER text sentiment for the top 5 comments for a post (if available).     
            *Rationale:*  Posts that gather comments quickly and have a highly positive or negative sentiment are likely to be related to popularity. 
        ''')
    with feature_table.expander("Average SBERT Sentence Embeddings of the Comments"):
        st.markdown('''
            *Description:*  We used the SBERT library to encode the text of of the top 5 comments for a post (if available)     
            *Rationale:*  NEEDS UPDATE the rich meaning from language encoded via SBERT, which is based on the state of the art BERT language model.
        ''')

st.subheader("") #create blank space
st.subheader("Feature Engineering (Option 2)")
feature_table2 = st.container()
with feature_table2:
    st.write("After experimentation on our scraped dataset we decided upon the following features:")
    f_col1, f_col2 = st.columns([1,2])
    f_col1.info("Number of comments per hour")
    f_col2.write('''
            *Description:*  This is a count of the comments each post has received, divided by the number of hours that have elapsed since the post was created.  
            *Rationale:*  Our research and intuition told us that the number of people commenting on a post is an indicator of likely popularity.
        ''')
    f_col1, f_col2 = st.columns([1,2])
    f_col1.info("Author Karma for the Post")
    f_col2.write('''
            *Description:*  We tracked the karma  of both comment and post authors at the time of making either a post or a comment.  
            *Rationale:*  Whilst people who have high Karma scores aren't necessarily ‘influencers’ in the normal social media sense of the word, their karma scores are a good proxy for this.  Our EDA looked to see if posts that were posted by ‘high karma’ authors were more likely to become popular as a result and whilst the correlation was surprisingly low we took this feature forward to the modeling stage to test if this contained any ‘signal’ for our predictive task.
        ''')
    f_col1, f_col2 = st.columns([1,2])
    f_col1.info("Hour and Day the Post was created")
    f_col2.write('''
            *Description:*  We recorded the hour that a post was made (UTC) to see the correlation with post popularity.  In our pipeline we ‘one hot’ encoded these features before passing them to our training/inference models.  
            *Rationale:*  These features have shown predictive power in other social media analytics tasks [2]. 
        ''')

st.subheader("") #create blank space
st.subheader("Feature Engineering (Option 3)")
feature_table3 = st.container()
with feature_table3:
    st.write("After experimentation on our scraped dataset we decided upon the following features:")
    f_col1, f_col2 = st.columns([1,2])
    f_col1.info("Number of comments per hour")
    f_col2.info('''
            *Description:*  This is a count of the comments each post has received, divided by the number of hours that have elapsed since the post was created.  
            *Rationale:*  Our research and intuition told us that the number of people commenting on a post is an indicator of likely popularity.
        ''')
    f_col1, f_col2 = st.columns([1,2])
    f_col1.info("Author Karma for the Post")
    f_col2.info('''
            *Description:*  We tracked the karma  of both comment and post authors at the time of making either a post or a comment.  
            *Rationale:*  Whilst people who have high Karma scores aren't necessarily ‘influencers’ in the normal social media sense of the word, their karma scores are a good proxy for this.  Our EDA looked to see if posts that were posted by ‘high karma’ authors were more likely to become popular as a result and whilst the correlation was surprisingly low we took this feature forward to the modeling stage to test if this contained any ‘signal’ for our predictive task.
        ''')
    f_col1, f_col2 = st.columns([1,2])
    f_col1.info("Hour and Day the Post was created")
    f_col2.info('''
            *Description:*  We recorded the hour that a post was made (UTC) to see the correlation with post popularity.  In our pipeline we ‘one hot’ encoded these features before passing them to our training/inference models.  
            *Rationale:*  These features have shown predictive power in other social media analytics tasks [2]. 
        ''')









# st.subheader("Related Work")
# st.write('''
#     placeholder text introducing the following tables of work we reviewed
#     ''')
# c = st.container()
# st.write("this will show last")
# c.write("paper 1")
# c.write("paper 2")

# col1, col2, col3 = st.columns(3)
# col1.write("**Paper Title**")
# col2.write("**Topic**")
# col3.write("**Relevance to project**")
# col1.write("[3] Reddit predictions")
# col2.write("Supervised Learning approaches to predict popularity")
# col3.write("This is a prediction task on similar data.")
# c.write("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse ")

# st.header("") #create blank space
# st.subheader("Related Work v2")
# st.write('''
#     placeholder text introducing the following tables of work we reviewed
#     ''')
# c = st.container()
# col1, col2, col3 = c.columns(3)
# col1.write("**Paper Title**")
# col2.write("**Topic**")
# col3.write("**Relevance to project**")
# col1.write("[3] Reddit predictions")
# col2.write("Supervised Learning approaches to predict popularity")
# col3.write("This is a prediction task on similar data.")
# col1.write("[3] Reddit predictions")
# col2.write("Supervised Learning approaches to predict popularity")
# col3.write("This is a prediction task on similar data.")
# c.write("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse ")
# st.write("placeholder for text at the end of the literature review")
# st.header("") #create blank space
# st.write('''
#     placeholder text introducing the following tables of work we reviewed
#     ''')
# c = st.container()
# col1, col2, col3 = c.columns(3)
# with c:
#     col1.info("**Paper Title**")
#     col2.info("**Topic**")
#     col3.info("**Relevance to project**")

#     col1.info("[3] Reddit predictions")
#     col2.info("Supervised Learning approaches to predict popularity")
#     col3.info("This is a prediction task on similar data.")
#     col1.info("[3] Reddit predictions")
#     col2.info("Supervised Learning approaches to predict popularity")
#     col3.info("This is a prediction task on similar data.")
#     c.info("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse) ")
#     with c.expander("See implications for our project"):
#         st.warning("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse ")









st.markdown("") #create blank line
st.header("Testing Reddit Access")
st.markdown("Testing Reddit access...")
# ### SECRETS TO DELETE ###
# REDDIT_USERNAME= 'Qiopta'
# REDDIT_PASSWORD= 'd3.xr@ANTED#2-L'
# APP_ID= '8oUZoJ3VwfzceEInW3Vd1g'
# APP_SECRET= 'pRg3qU2brsbsyPrPaNP26vxPgwAJbA'
# APP_NAME= 'Capstone2'
# ### SECRETS TO DELETE ###

REDDIT_USERNAME= st.secrets['REDDIT_USERNAME']
REDDIT_PASSWORD= st.secrets['REDDIT_PASSWORD']
APP_ID= st.secrets['APP_ID']
APP_SECRET= st.secrets['APP_SECRET']
APP_NAME= st.secrets['APP_NAME']

reddit = praw.Reddit(
    client_id       = APP_ID,
    client_secret   = APP_SECRET,
    user_agent      = APP_NAME, 
    username        = REDDIT_USERNAME,  
    password        = REDDIT_PASSWORD,  
    check_for_async = False # This additional parameter supresses some annoying warnings about "asynchronous PRAW " https://asyncpraw.readthedocs.io/en/stable/
)


if st.button("Get new posts"):
    for submission in reddit.subreddit("investing").new(limit=5):
        if submission.author==None or submission.author=="Automoderator":
            continue
        else:
            # st.markdown("Post ID:")
            # st.text(submission.id)
            # st.markdown("Post Title:")
            # st.text(submission.title)
            st.markdown(f"__Post ID:__ {submission.id} __// Post Title:__ {submission.title} ")



st.title("End Blog Post Draft")



st.header("Testing Interactivity")
st.markdown("> Just for fun, enter a number and choose a team member to see what happens...")








#  This is equivalent to <input type = "number"> in HTML.
# Input bar 1
a = st.number_input("Enter a Number")

# # Input bar 2
# b = st.number_input("Input another Number")

# This is equivalent to the <select> tag for the dropdown and the <option> tag for the options in HTML.
# Dropdown input
names = st.selectbox("Select Team Member", ("Erik", "Chris","Antoine"))

# put it in an if statement because it simply returns True if pressed. This is equivalent to the <button> tag in HTML.
# If button is pressed
if st.button("Submit"):
    
    # # Unpickle classifier
    # clf = joblib.load("clf.pkl")
    
    # # Store inputs into dataframe
    # X = pd.DataFrame([[height, weight, eyes]], 
    #                  columns = ["Height", "Weight", "Eyes"])
    # X = X.replace(["Brown", "Blue"], [1, 0])
    
    # # Get prediction
    # prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"{names} just won {a} dollars!!!")
    # Note that print() will not appear on a Streamlit app.
    st.markdown(f"{names} just won {a} dollars!!!")


#st.markdown renders any string written using Github-flavored Markdown. It also supports HTML but Streamlit advises against allowing it due to potential user security concerns.

st.header("Project Start")
st.subheader("In Introduction to our Project")
st.markdown("But seriously, we're here to talke about our blog.  This might be how text will appear in our blog.")





st.subheader("A Code Block")
# st.code renders single-line as well as multi-line code blocks. There is also an option to specify the programming language.
st.code("""
def Team_Augury_feature_functions(df):
    df = df.copy
    df['column'] = df['old_column'].apply(lambda x: 1 if True else 0, axis1)
    return None
""", language="python")


