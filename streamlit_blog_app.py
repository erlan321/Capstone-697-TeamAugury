import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
#import joblib
#from sklearn.linear_model import LogisticRegression
from PIL import Image
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from profanity_filter import ProfanityFilter
from functions import Team_Augury_blog_praw_functions
from functions import Team_Augury_blog_hpt_charts
from functions import Team_Augury_feature_functions
import spacy  #needed for language profanity filtering?
#spacy.load('en')
import pickle
import altair as alt

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
    Reddit is a popular social media networking site, and a subreddit is a forum on Reddit that is dedicated to a specific topic.  Project Augury is focussed on exploring what makes a post on these subreddits to be popular.  Our initial background research suggested that predictive tasks on Reddit  work better in thematic subreddits and do not necessarily generalize from one theme or subreddit to another.    But rather than focus on a single subreddit, we decided to investigate a small group of four subreddits related to a single theme: investing.  

     - r/investing
     - r/wallstreetbets
     - r/StockMarket
     - r/stocks

    We chose the investing theme because it felt particularly topical given the widely reported “GameStop” incident in 2021.  In this incident, the subreddit r/wallstreetbets achieved some notoriety where a formerly little known trader named Keith Gill achieved gains of $40M plus in a short period of time referred to as the ‘Reddit Rally’ [1].  We chose those four subreddits because they appear to be the four most active forums related to the theme according to the website [subredditstats.com](https://subredditstats.com) based on an analysis of subscribers and posts per day.  
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
        **Title:** Data Stories. We analyzed 4 million data points to see what makes it to the front page of reddit. Here’s what we learned.[2]  
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
    There are clearly ethical implications relating from broadcasting messages on social media and the related investments to which these messages refer, as highlighted by the ‘Reddit Rally’ described above. In project Augury we are only looking at the popularity of posts, and we have not correlated this to market activity, which could be an extension of this work. To some extent this research has already been investigated by Muxi Xu  in 2021 [13] and also by Hu et al [14]. Therefore we have made no further mitigations in our project related to market ethics.  
    
    Social Media is often thought of as an open and public forum of discussion.  But an important ethical consideration of any data science project related to social media is that saying something in “public” may not necessarily mean “consent” to using a person’s name or username in published research [15].  While we did not have any need for usernames in the analysis we were performing, we did gather usernames into our database in order to track certain features of each Reddit post, and we are aware that if we were to ever use those usernames in future work, our database would could provide us with an already anonymized id number to protect each individuals’ privacy.  
    
    A different aspect of  the free and open nature of social media, and Reddit in particular, is that it can  lead to environments which become toxic in a number of ways.  To ameliorate this we did take some action in our post-processing of our database.  The subreddits which are looking at Investment are typically more professional in nature so the main way in which toxicity occurs in these fora is through the use of profane language. Thus, in our pipeline we have removed this language from our analysis using the python module profanity-filter 1.3.3 which replaces profane words with the character “*”.  

    ''')


st.subheader("") #create blank space
st.header("Our Project Workflow")
st.write("The below graphic illustrates our project Augury workflow.  Below we will provide more details on each component of the workflow.")
project_pipeline_image = Image.open('blog_assets/project_pipeline2.png')
st.image(project_pipeline_image, caption='Project Augury Workflow')

st.subheader("") #create blank space
st.subheader("Scraping Reddit Data") 
st.write('''
In order to request data from Reddit’s API, we decided to use the widely known PRAW python library to do our data scraping.  Our approach to scraping was informed primarily by studying the PRAW online documentation [(docs)](https://praw.readthedocs.io/en/stable/).    

Because we were starting from a “blank slate” in our domain knowledge about Reddit data, we knew we wanted to track each post over a 24 hour period in order to effectively perform Exploratory Data Analysis (EDA) about how we would approach our prediction task.  
We knew we wanted to see a posts popularity over time in order to see how and when popular posts developed, so we designed a scraping pipeline that captured a sample of ‘new’ posts each hour and stored it in our database.  

Our initial scraping process sought to scrape information about the five newest posts per hour from our selection of subreddits, and included information about the top 5 comments for each post  (more details about the features is below).  However, we discovered that this amount of data caused a “timeout” problem with the AWS Lambda service we were using, so by March 1st, 2022 we had to decide to scale back the number of new posts being scraped each hour to one per subreddit.  

**Pre-storage Cleaning:**
Despite having to reduce the original pace of data scraping, one of the benefits of our scraping process using PRAW was we were able to scrape and store just the data that we wanted, and we were able to filter out unwanted posts and comments, such as those that had already been deleted for any reason or that were authored by a subreddit Automoderator.  In that sense, the data we were gathering hourly was “clean” and usable right away without additional filtering when the data came back out from our database.  
**Post-storage Cleaning:**
As we noted above, we did apply one important additional post-scraping process to our data when extracting from our database, which was to remove profanity.
    ''')

st.subheader("") #create blank space
st.subheader("Storing our Data")
st.write('''
    We stored our scraped Reddit data in a PostGreSQL instance on AWS.  Our database design/schema is intended to reduce duplication to a minimum, thereby optimizing the functioning of the relational database and minimize storage.  We also designed this database with a view that we may want to perform additional projects on this data beyond what we achieve during this Capstone course, and we identify some of those areas of future work in our conclusion section below.  Our database design/schema is illustrated below:
    ''')
db_schema_image = Image.open('blog_assets/db_schema.png')
col1, col2, col3 = st.columns([1,5,1]) #column trick to center on the webpage
with col1:
    st.write("")
with col2:
    st.image(db_schema_image, caption='Augury Database Schema in AWS')
with col3:
    st.write("")


st.subheader("") #create blank space
st.subheader("Exploratory Data Analysis (EDA)")
st.write('''
    As mentioned above, we were successfully scraping Reddit consistently starting on March 1st of 2022.  We began performing Exploratory Data Analysis (EDA) with the data we had gathered so far on March 28th of 2022, which represented roughly 1,200 posts tracked over a 24 hour period.  
    
    Our objective is to predict whether or not a post will become “popular”.  A “popular” post is one that would be near the top of the subreddit when a user goes to the webpage or opens the app (in Reddit’s vocabulary, the post is “Hot”).  From perusing Reddit’s own message boards, we understand that a rough approximation for this measure of “popularity” is:
    ''')
st.latex(r"popularity = \frac{upvotes}{hours}")
st.write('''
    We illustrate our exploration of “popularity” in the below chart on a sample of our data, where the faint blue lines are the popularity of individual posts scraped hourly for 24 hours.  Overlaid on this chart are three lines representing the 50th percentile (half the data), 80th percentile (the top quintile), and 90th percentile (the top decile).  
    ''')
popularity_image = Image.open('blog_assets/EDA_popularity.png')
st.image(popularity_image, caption='Proxy for Post Popularity')
st.write('''
    From this chart, we make two decisions about our prediction task.  First, that popularity can peak at different times, but using a post’s popularity at hour 3  seems to be appropriate for our prediction variable.  Second, we see a lot of the subsample achieves very low popularity, so we feel comfortable using a threshold for “popular” close to the top Quintile (the red line), or a “popularity” value of 10.  Thus, the determination of “popular” versus “not popular” in our prediction task is determined by if a post has a “popularity proxy” of over or under 10 by hour 3.  
    ''')
st.write('''
    In order to get a sense of how some of the basic information we have collected about the posts (and the comments related to each post) might be influencing popularity on Reddit, we looked at a correlation analysis against our proxy measure of popularity.  We see that both the total number of comments (normalized for how old the post is) as well as the number of upvotes those comments receive have a strong positive correlation to popularity.  On the other hand, data related to a posts' author karma and commenters' karma seems to have a very weak relationship to popularity.    

    ''')
corr_image = Image.open('blog_assets/eda_basic_corr.png')
col1, col2, col3 = st.columns([1,5,1]) #column trick to center on the webpage
with col1:
    st.write("")
with col2:
    st.image(corr_image, caption='Correlation of Basic Data to Popularity')
with col3:
    st.write("")

st.write('''
    We also looked at what relationships might exist between post popularity and the _time_ and _day_ that post was created.  To look at this, we looked at what the maximum popularity each post in the sample achieved during a 24 hour period and calculated the median for each hour of the day and the day of the week that the post was created.  We should note first that we scraped and stored this data in UTC time, and made no adjustment to a different time zone in the visualization.    
    Given the global nature of online communities such as Reddit and that fact that the data show was scraped and stored in UTC time, we don't want to read much into these relationships, but we do notice some interesting differences in popularity based on when the post is created.  For instance, in our sample we see a higher median maximum popularity for posts created on the weekend (Friday, Saturday, Sunday).  This makes some intuitive sense.  In regard to what hour of the day the post was created, we would not say there is any strong or consistent trend within our sample about what hour the post is created, though in future research it could be interesting to dig into some of the spikes we see in the chart.  

    ''')
temporal_image = Image.open('blog_assets/eda_temporal_chart.png')
st.image(temporal_image, caption='Analysis the median maximum popularity achieved based on the day or time a post is created')
st.write('''
    In summary, the above EDA was very helpful in determining our classification of "popular" versus "not popular", and also gave us some initial expectations about the feature choices for our model, for which we provide our full rationale in the next section below.
    ''')
    

st.subheader("") #create blank space
st.subheader("Feature Engineering")
feature_table = st.container()
with feature_table:
    st.write("Through a combination of our EDA, our review of Related Works, our intuition, and our understanding of Reddit, we chose to engineer the following features for use in our prediction task.  (*Click on a feature for description & rationale*)")
    with feature_table.expander("Number of comments per hour"):
        st.markdown('''
            *Description:*  This is a count of the comments each post has received, divided by the number of hours that have elapsed since the post was created.  

            *Rationale:*  Our research and intuition told us that the number of people commenting on a post is an indicator of likely popularity.
        ''')
    with feature_table.expander("Author Karma for the Post"):
        st.markdown('''
            *Description:*  We tracked the karma of both comment and post authors at the time of making either a post or a comment.  Reddit karma is a score a user recieves based on the user's activity.  

            *Rationale:*  Whilst people who have high karma scores aren't necessarily ‘influencers’ in the normal social media sense of the word, their karma scores are a good proxy for this.  Our EDA looked to see if posts that were posted by ‘high karma’ authors were more likely to become popular as a result and whilst the correlation was surprisingly low we took this feature forward to the modeling stage to test if this contained any ‘signal’ for our predictive task.
        ''')
    with feature_table.expander("Hour and Day the Post was created"):
        st.markdown('''
            *Description:*  We recorded the hour that a post was made (UTC) to see the correlation with post popularity.  In our pipeline we ‘one hot’ encoded these features before passing them to our training/inference models.  
            
            *Rationale:*  These features have shown predictive power in other social media analytics tasks [2]. 
        ''')
    with feature_table.expander("VADER Text Sentiment of the Post"):
        st.markdown('''
            *Description:*  We used the VADER sentiment library to classify the sentiment of each posts text.  This produced a value in the range of -1, +1.  [(docs)](https://github.com/cjhutto/vaderSentiment)     

            *Rationale:*  We believe text that has a strongly positive or negative sentiment is more likely to become popular than something that is neutral. 
        ''')
    with feature_table.expander("SBERT Sentence Embeddings of the Post"):
        st.markdown('''
            *Description:*  We used the SBERT library to encode the text of both Posts and Comments.  The SBERT interface was simple to use and produced in effect 380+ features for our classifiers for each post.  [(docs)](https://www.sbert.net/docs/pretrained_models.html)     

            *Rationale:*  NEEDS UPDATE the rich meaning from language encoded via SBERT, which is based on the state of the art BERT language model.
        ''')
    with feature_table.expander("Average Upvotes for the Top 5 Comments on the Post (per Hour)"):
        st.markdown('''
            *Description:*  We look at the top 5 comments for a post (if available) and see how many upvotes that comment has gotten.     
            
            *Rationale:*  Posts that gather comments quickly will likely have their popularity influenced by upvotes to those comments. 
        ''')
    with feature_table.expander("Average Author Karma for the Top 5 Comments on the Post"):
        st.markdown('''
            *Description:*  We look at the Commentor Karma for the top 5 comments for a post (if available).  Reddit karma is a score a user recieves based on the user's activity.   

            *Rationale:*  Posts that gather comments quickly from authors with a high reputation should impact the Post's popularity. 
        ''')
    with feature_table.expander("Average VADER Text Sentiment of the Top 5 Comments of the Post"):
        st.markdown('''
            *Description:*  We look at the average VADER text sentiment for the top 5 comments for a post (if available).  [(docs)](https://github.com/cjhutto/vaderSentiment)     

            *Rationale:*  Posts that gather comments quickly and have a highly positive or negative sentiment are likely to be related to popularity. 
        ''')
    with feature_table.expander("Average SBERT Sentence Embeddings of the Comments"):
        st.markdown('''
            *Description:*  We used the SBERT library to encode the text of of the top 5 comments for a post (if available)  [(docs)](https://www.sbert.net/docs/pretrained_models.html)     

            *Rationale:*  NEEDS UPDATE the rich meaning from language encoded via SBERT, which is based on the state of the art BERT language model.
        ''')

# st.subheader("") #create blank space
# st.subheader("Feature Engineering (Option 2)")
# feature_table2 = st.container()
# with feature_table2:
#     st.write("After experimentation on our scraped dataset we decided upon the following features:")
#     f_col1, f_col2 = st.columns([1,2])
#     f_col1.info("Number of comments per hour")
#     f_col2.write('''
#             *Description:*  This is a count of the comments each post has received, divided by the number of hours that have elapsed since the post was created.  
#             *Rationale:*  Our research and intuition told us that the number of people commenting on a post is an indicator of likely popularity.
#         ''')
#     f_col1, f_col2 = st.columns([1,2])
#     f_col1.info("Author Karma for the Post")
#     f_col2.write('''
#             *Description:*  We tracked the karma  of both comment and post authors at the time of making either a post or a comment.  
#             *Rationale:*  Whilst people who have high Karma scores aren't necessarily ‘influencers’ in the normal social media sense of the word, their karma scores are a good proxy for this.  Our EDA looked to see if posts that were posted by ‘high karma’ authors were more likely to become popular as a result and whilst the correlation was surprisingly low we took this feature forward to the modeling stage to test if this contained any ‘signal’ for our predictive task.
#         ''')
#     f_col1, f_col2 = st.columns([1,2])
#     f_col1.info("Hour and Day the Post was created")
#     f_col2.write('''
#             *Description:*  We recorded the hour that a post was made (UTC) to see the correlation with post popularity.  In our pipeline we ‘one hot’ encoded these features before passing them to our training/inference models.  
#             *Rationale:*  These features have shown predictive power in other social media analytics tasks [2]. 
#         ''')

# st.subheader("") #create blank space
# st.subheader("Feature Engineering (Option 3)")
# feature_table3 = st.container()
# with feature_table3:
#     st.write("After experimentation on our scraped dataset we decided upon the following features:")
#     f_col1, f_col2 = st.columns([1,2])
#     f_col1.info("Number of comments per hour")
#     f_col2.info('''
#             *Description:*  This is a count of the comments each post has received, divided by the number of hours that have elapsed since the post was created.  
#             *Rationale:*  Our research and intuition told us that the number of people commenting on a post is an indicator of likely popularity.
#         ''')
#     f_col1, f_col2 = st.columns([1,2])
#     f_col1.info("Author Karma for the Post")
#     f_col2.info('''
#             *Description:*  We tracked the karma  of both comment and post authors at the time of making either a post or a comment.  
#             *Rationale:*  Whilst people who have high Karma scores aren't necessarily ‘influencers’ in the normal social media sense of the word, their karma scores are a good proxy for this.  Our EDA looked to see if posts that were posted by ‘high karma’ authors were more likely to become popular as a result and whilst the correlation was surprisingly low we took this feature forward to the modeling stage to test if this contained any ‘signal’ for our predictive task.
#         ''')
#     f_col1, f_col2 = st.columns([1,2])
#     f_col1.info("Hour and Day the Post was created")
#     f_col2.info('''
#             *Description:*  We recorded the hour that a post was made (UTC) to see the correlation with post popularity.  In our pipeline we ‘one hot’ encoded these features before passing them to our training/inference models.  
#             *Rationale:*  These features have shown predictive power in other social media analytics tasks [2]. 
#         ''')

st.subheader("") #create blank space
st.header("Modeling Inferences & Evaluation")
st.subheader("Model Candidates")
st.markdown('''
	We selected three Supervised Learning models on which to attempt our Reddit post popularity prediction.  We kept our model exploration within the popular Scikit-Learn python library we learned in class.
	 - Logistic Regression (LR):  LR was chosen as it also appeared in some of the related work that we reviewed, and this simple classifier is often used as a “baseline” prediction model.  [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
	 - Support Vector Classification (SVC): SVC was chosen as it also appeared in the related papers that were pursuing similar goals to our Augury project.  [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)  
	 - Gradient Boosting Classifier (GBDT):  GBDT was chosen based on our intuition and also feedback from our professors that Gradient Boosted classifiers are often winning data science competitions lately.  [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
	''')


st.subheader("Hyperparameter Tuning")
st.write('''
    In order to perform hyperparameter tuning for these three models, we explored a few different options to perform cross-validation.  We ultimately decided on Scikit-Learn's GridSearchCV method [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).  
    We also explored RandomizedSearchCV and Nested Cross-Validation.  Since our dataset was relatively small the efficiency of RandomizedSearchCV was not required, and this also would have made the Nested Cross-Validation difficult due to too small samples.  
    Importantly, we chose to use the standard _cv_ object within GridsearchCV as it implements the stratified k-fold approach, thus ensuring we do not split without the much smaller “popular” class.  

    ''')
kfold_image = Image.open('blog_assets/StratifiedKFoldCVExplainerGraph.png')
st.image(kfold_image, caption='Illustration of how Stratified K-Fold Cross Validation works in Project Augury')
st.write("") #blank space
st.write('''
    **Project train / validation / test split:**  
    Related to the above illustration of how we are cross-validating our model choices we want to highlight what this implies for our project's split between training, validation, and test data sets.  
    We have been successfully scraping from Reddit since March 1st of 2022, and we are going to be training and validating our model choices on data up until April 18th of 2022, or 48 days of hourly scraping.  Therefore, our "unseen" test set of data will be from April 18th to XXXXXXX of 2022, or XXXXXXX days.  
    
    _Here is how these splits translate into the percentages of our overall data and number of individual posts:_  

    ''')
col1, col2, col3 = st.columns(3)
col1.info("**Training**")
col2.info("**Validation**")
col3.info("**Testing**")
col1.write("68% of data")
col2.write("17% of data")
col3.write("15% of data")
col1.write("~2,200 posts")
col2.write("~560 posts")
col3.write("~500 posts")
st.write("!! UPDATE NUMBERS ABOVE BEFORE SUBMISSION !!")

st.write("") #blank space
st.write('''
    **Tuning Iterations:**  
    We iterated through roughly 1,000 combinations of parameters across LR, SVC, and GBDT models, for which we needed to develop both objective and subjective methods of making parameter choices.  
    ''')
st.write('''
    _LR Tuning:_ For LR we flipped between the _solver_ parameter, the types of penalty, and a range of values of C we've seen before in our course work.  
    _Parameter dictionary used in our code_
    ''')
st.code('''
    parameters = [{"clf__C":np.logspace(-5, 10, num=16, base=2),
                "clf__solver":["liblinear", "lbfgs"],
                "clf__penalty":["l1", "l2"],
                },
                {"clf__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "clf__solver":["lbfgs"],
                "clf__penalty":["l2"],}]
    ''', language="python")
st.write('''
    _SVC Tuning:_ We looked at each of the four kernels individually to avoid errors that would make us lose a lot of compute time. The range for C and gamma was taken from A Practical Guide to Support Vector Classification by Chih-Wei Hsu et al.[16], these were slightly more thorough in their tests than other information available.  
    _Parameter dictionary used in our code (approximately)_
    ''')
st.code('''
    parameters = {"clf__C":np.logspace(-5, 15, num=21, base=2), 
                "clf__kernel":["rbf","linear","poly","sigmoid"],
                "clf__degree": [3, 4, 5], #only related to the polynomial kernel
                'clf__gamma': np.logspace(-15, 3, num=19, base=2)}
    ''', language="python")
st.write('''
    _GBDT Tuning:_ We tuned the GradiantBoostingClassifier on the following parameters. These were picked from great recommendations from MachineLearningMastery [17] as well as some adjustments for speed (reducing the number of total folds) and introducing subsamples for Stochastic Gradient Descent as per the recommendation of the Hastie et al. 2009 [CITATION NEEDED] paper explained in the scikit learn documentation.  
    _Parameter dictionary used in our code_
    ''')
st.code('''
    parameters = {"clf__learning_rate":[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                "clf__n_estimators":[50,75,100,125,150,175,200, 250],
                "clf__max_depth":[3,5,8],
                "clf__max_features":["log2","sqrt"],
                "clf__subsample":[0.5,0.75,1.0],}
    ''', language="python")
    
st.write("") #blank space
st.write(''' 
    **Hyperparameter Decision Process:**  
    With over 1,000 different options produced from our hyperparameter tuning process, we used both objective and subjective methods of coming to a decision about which hyperparameters were most appropriate, and thus which of the three styles of model to ultimate use for our prediction task.  
    
    For each iteration, we output the F1 Score calculation for each model, both on the training data and validation data.  F1 Score felt most appropriate as it has a better ability to account for imbalanced classification data such as ours, where only a small percentage of posts would become "popular.  Also, the F1 Score was the metric (and sometimes sole metric) used similar projects we saw in related works.  Using this information, we went through these steps in each notebook:
     - We used Jupyter Notebooks to visualize the results, which we felt was the most appropriate tool for this process.
     - We began by viewing all of the F1 calculations for each model, as messy as it was, and noted any observable patterns.
     - We then created charts that looked at each hyperparameter, looking for levels of the hyperparameter that gave higher performance "all else equal".  By this we mean we took the median performance metric for each level of an individual hyperparameter.  From this method we were able to see general trends in the individual hyperparameters that resulted in higher or lower model performance.
     - Our _"objective"_ decision was to favor levels of a hyperparameter that yielded better F1 scores.
     - Our _"subjective"_ decision was to favor levels of a hyperparameter that were neither at the extremes of our tuning ranges, nor resulted in results that looked "too good" and might indicate over-fitting.  
    
    ''')
st.write("") #blank space
st.write('''
    **Visualizing the Hyperparameter Decision Process:**  
    The below charts illustrate the process we described in the bullet-points above.  Choose either F1 Score or Accuracy Score to visualize the median results across the different models and parameters for yourself.  
    ''')
#############################################################
score_metric = st.selectbox("Select the Scoring Metric to view a sample of our hyperparameter tuning visualizations:", ("F1","Accuracy"))
st.write("") #blank space
st.altair_chart(Team_Augury_blog_hpt_charts.hpt_lr_chart( score_metric), use_container_width=False)
st.altair_chart(Team_Augury_blog_hpt_charts.hpt_svc_chart( score_metric), use_container_width=False)
st.altair_chart(Team_Augury_blog_hpt_charts.hpt_gbdt_chart( score_metric), use_container_width=False)
##############################################################
st.write(''' 
    **Hyperparameter Decisions:**  
    The above process resulted in the following hyperparameter decisions:
     > **LR** (_matches the default settings_):  
     >> Solver: lbfgs   
     >> Penalty: L2  
     >> C: 1.0  
     > **SVC**:  
     >> Kernel: rbf    
     >> C: 0.125  
     >> Gamma: 0.0078125  
     > **GBDT**:  
     >> Learning_rate: 0.15  
     >> N_estimators: 150  
     >> Max_depth: 3  
     >> Max_features: sqrt  
     >> Subsample:  0.5  

    ''')

st.write('''
    Performance metrics of the above "tuned" models:
    ''')
hpt_df = pd.DataFrame(data={
    'Tuned_Model': ['LR','SVC','GBDT'],
    'F1_Score_Training': [0.76,0.73,0.94],
    'F1_Score_Validation': [0.39,0.52,0.35],
    'Accuracy_Training': [0.91,0.75,0.98],
    'Accuracy_Validation': [0.77,0.73,0.79],
    })
st.table(hpt_df)


st.subheader("Model Choice")

st.subheader("Feature Importance of the Model")

st.subheader("Model Performance (on unseen data)")

st.subheader("Real-Time Model Prediction")

#load pkl'd classifier (clf)
#filename = "models/SVC_final_model.pkl" 
filename = "models/LogisticRegression_final_baseline_model.pkl" 
clf = pickle.load(open(filename, 'rb'))

st.markdown('''
    We wanted to give the readers of this blog the opportunity to see our model's predictions on the current data in Reddit!  
     - Because our model was generalized around only a single theme of "investing", we will limit you to the four subreddits we used in our study described above.  
     - Below we give you an opportunity to choose either all or some of those four investing subreddits, and also the number of posts you want a recommendation for from each subreddit.  
     - This process will only scrape posts from Reddit that have been created within the last hour, similar to our project, so _you may see less posts than you are attempting to scrape_.

    ''')


#st.markdown("Testing Reddit access...")
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
### Set up PRAW variables
#subreddit_scrape_list = ["investing", "wallstreetbets", "StockMarket", "stocks",]
subreddit_scrape_list = []
subreddit_selection_list = st.multiselect(
    "Select either all subreddits about investing, or just a selection of subreddits:", 
    ["r/investing","r/wallstreetbets","r/StockMarket","r/stocks"],
    ["r/investing","r/wallstreetbets","r/StockMarket","r/stocks"],
    )
for item in subreddit_selection_list:
    subreddit_scrape_list.append(item[2:]) #removes the r/
subreddit_scrape_list = subreddit_scrape_list
#st.write("You selected:", subreddit_scrape_list)

#n_posts = 5
n_posts = st.selectbox("Select the number of posts you'd like to scrape from each subreddit:", (1,3,5))

n_posts = int(n_posts)
n_comments = 5 
hrs_to_track = 1 #number of hours to track a post/submission
#time_of_batch = datetime.utcnow().replace(microsecond=0)                                      
char_limit = 256 #character limit for the text fields in the database


st.write("") #blank space
if st.button("Test creating PRAW df for our pipeline"):
    st.write("Number of posts we'll try to collect:", n_posts*len(subreddit_scrape_list))

    time_of_batch = datetime.utcnow().replace(microsecond=0)
    new_submission_list = Team_Augury_blog_praw_functions.blog_submission_list(reddit=reddit, time_of_batch=time_of_batch, hrs_to_track=hrs_to_track, n_posts=n_posts, subreddit_scrape_list=subreddit_scrape_list)
    post_data, comments_data = Team_Augury_blog_praw_functions.blog_scrape_dataframes(reddit=reddit, time_of_batch=time_of_batch, n_comments=n_comments, char_limit=char_limit, new_submission_list=new_submission_list)
    feature_df = Team_Augury_blog_praw_functions.blog_feature_creation(post_data, comments_data)
    st.table(feature_df)
    df = feature_df[['sr','post_id','post_text']].copy()
    st.write("df",df)

    st.subheader("") #blank space
    feature_df = Team_Augury_blog_praw_functions.blog_X_values(feature_df)
    st.write("X_values for pkl'd model")
    st.table(feature_df)
    st.write("len(feature_df.columns):",len(feature_df.columns))

    predictions = clf.predict(feature_df)
    st.write("predictions...", predictions)
    prediction_probas = clf.predict_proba(feature_df)
    prediction_probas.rename({0:'Non_Popular_Probability', 1:'Popular_Probability'}, axis=1)
    st.write("prediction probabilities...", prediction_probas)
    st.write("temp", prediction_probas['Popular_Probability'])
    # df = pd.concat([df, prediction_probas], axis=0)
    # st.write("output df",df)







# if st.button("Get new posts"):
#     for submission in reddit.subreddit("investing").new(limit=5):
#         if submission.author==None or submission.author=="Automoderator":
#             continue
#         else:
#             # st.markdown("Post ID:")
#             # st.text(submission.id)
#             # st.markdown("Post Title:")
#             # st.text(submission.title)
#             st.markdown(f"__Post ID:__ {submission.id} __// Post Title:__ {submission.title} ")

st.header("Conclusions & Future Work")



st.header("Appendix: Statement of work")
st.write('''
The team worked across the entire project, but the below highlights the areas each team member either focussed, led or made a major contribution:  

Antoine Wermenlinger led on the Supervised Learning aspect of the project from the long list of models, through to the down select and hyperparameter tuning. He initiated the feature Engineering work and was the originator of the project's core goal - to predict popularity of Reddit posts.  

Chris Lynch led on the instantiation of our AWS instance, account and functions, notably the Lambda Function, Event Bridge Scheduling and the creation of our RDS (a postgreSQL database).  In the early phases of the project Chris led on the Reinforcement Learning aspects of the project creating a Reddit Simulator environment and a Q learning model, which eventually we decided not to use. Chris initiated the draft for our team Blog post.  

Erik Lang led on the Social Media Scraping and analytics aspects of the project, creating the PRAW based code that scraped and cleaned the raw Reddit data that was written into our postgreSQL instance. He made major contributions to the feature engineering and team standups.   Finally Erik created our Streamlit instance, that is the host site for this Blog. 
    ''')



st.header("Appendix: References")

st.header("!!! END OF BLOG POST !!!")











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

















# st.header("Testing Interactivity")
# st.markdown("> Just for fun, enter a number and choose a team member to see what happens...")








# #  This is equivalent to <input type = "number"> in HTML.
# # Input bar 1
# a = st.number_input("Enter a Number")

# # # Input bar 2
# # b = st.number_input("Input another Number")

# # This is equivalent to the <select> tag for the dropdown and the <option> tag for the options in HTML.
# # Dropdown input
# names = st.selectbox("Select Team Member", ("Erik", "Chris","Antoine"))

# # put it in an if statement because it simply returns True if pressed. This is equivalent to the <button> tag in HTML.
# # If button is pressed
# if st.button("Submit"):
    
#     # # Unpickle classifier
#     # clf = joblib.load("clf.pkl")
    
#     # # Store inputs into dataframe
#     # X = pd.DataFrame([[height, weight, eyes]], 
#     #                  columns = ["Height", "Weight", "Eyes"])
#     # X = X.replace(["Brown", "Blue"], [1, 0])
    
#     # # Get prediction
#     # prediction = clf.predict(X)[0]
    
#     # Output prediction
#     st.text(f"{names} just won {a} dollars!!!")
#     # Note that print() will not appear on a Streamlit app.
#     st.markdown(f"{names} just won {a} dollars!!!")


#st.markdown renders any string written using Github-flavored Markdown. It also supports HTML but Streamlit advises against allowing it due to potential user security concerns.

# st.header("Project Start")
# st.subheader("In Introduction to our Project")
# st.markdown("But seriously, we're here to talke about our blog.  This might be how text will appear in our blog.")





# st.subheader("A Code Block")
# # st.code renders single-line as well as multi-line code blocks. There is also an option to specify the programming language.
# st.code("""
# def Team_Augury_feature_functions(df):
#     df = df.copy
#     df['column'] = df['old_column'].apply(lambda x: 1 if True else 0, axis1)
#     return None
# """, language="python")


