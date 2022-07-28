import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from profanity_filter import ProfanityFilter
from functions import Team_Augury_blog_praw_functions
from functions import Team_Augury_blog_hpt_charts
from functions import Team_Augury_feature_functions
import spacy  #needed for language profanity filtering to work on streamlit
import pickle
import altair as alt


# Title
st.title("Project Augury: Predicting which Investing posts on Reddit are likely to become popular")
st.caption(" **augury** _noun_; a sign of what will happen in the future     -- Cambridge Dictionary")
st.markdown(">Resources:  \n>> Git Repository: [Github Link](https://github.com/erlan321/Capstone-697-TeamAugury)  \n>> Blog Post: [Streamlit Link](https://share.streamlit.io/erlan321/capstone-697-teamaugury/main/streamlit_blog_app.py)  \n>Authors:  \n>> Antoine Wermenlinger (awerm@umich.edu)  \n>> Chris Lynch (cdlynch@umich.edu)  \n>> Erik Lang (eriklang@umich.edu)")

st.header("Summary")
st.write('''
     - Project Augury aims to predict what posts are likely to become popular on the social media platform Reddit. It does this specifically by looking at four subreddits that are themed around investing.  
     - This project defines a new measure of popularity to include a temporal element: “Which post titles are likely to be popular within three hours of being posted?”  This temporal aspect is more specific and nuanced compared to previous related work.  
     - We found that using Natural Language Processing (NLP) based Sentence Bert (SBERT) encodings of post titles and underlying comments on the post gave the strongest predictive power, and trained a support vector machine classifier model to beat a baseline classifier in predicting what will become “popular”, and posts impressive performance relative to related work in the domain.  
     - This blog describes the background work of this project, the workflow we created for our analyses, and gives the reader a chance to experiment with live data from Reddit to see our model’s prediction on current posts on a subreddit.  

    ''')
st.markdown('''---''')
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
        **Title:** Predicting the Popularity of Reddit Posts with AI.[2]  
        **Topic:** Supervised Learning approaches to predict popularity    
        **Implication for our project:** This is a prediction task on similar data.  The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF (Term Frequency-Inverse Document Frequency), and LDA (Latent Dirichlet Allocation) trained on features extracted with Naive Bayes and SVM.  This work is based on regression, trying to predict the number of upvotes, whereas Augury aims to predict whether a post will be popular or not ( a classification problem) within a three hour window.   
        ''')
    related_work.info('''
        **Title:** Using Machine Learning to Predict the Popularity of Reddit Comments.[3]  
        **Topic:** Supervised Learning approaches to predicting Reddit comment Popularity    
        **Implication for our project:** Comparable aims, different NLP and looked only at Comments rather than posts.  Handling similar data, the author aimed to predict comment popularity.  Achieved relatively low accuracy scores ranging from 42% to 52.7% with a Decision Tree Classifier performing best. Cohen's Kappa statistic is applied to show results were, in fact, not much better than random.  In their conclusions they suggest research looks at temporal proximity of comments to posts, a key feature in Augury.
        ''')
    related_work.info('''
        **Title:** Popularity prediction of reddit texts.[4]  
        **Topic:** Supervised learning approach to predict Reddit post popularity    
        **Implication for our project:** Comparable objectives, uses different NLP and features. Focuses on using Topics to determine predictive task.   Achieved 60-75% accuracy on the task, using Latent Dirichlet Allocation (LDA) and Term Frequency Inverse Document Frequency (TFIDF) to classify topics in posts to explore the relationship between topics and posts in order to predict using Naive Bayes and Support Vector Machine Classifiers what will become popular. Augury includes topic modeling as a feature, and our initial model suite included these classifiers.  
        ''')
    related_work.info('''
        **Title:** Predicting the Popularity of Reddit Posts.[5]  
        **Topic:** Supervised learning approach to predict Reddit post popularity    
        **Implication for our project:** Conducted similar time of day, day of week features to Augury. Also performed sentiment analysis, with a different method. Finally treated the problem as a regression rather than classification one.  Our early experiments found classification to be better suited to our objective.
        ''')
    related_work.info('''
        **Title:** Deepcas: An end-to-end predictor of information cascades.[6]  
        **Topic:** Neural Network approach to predicting information cascades    
        **Implication for our project:** The prediction task in DeepCas was quite different to Augury. The problem definition included a Markov Decision Process as a ‘deep walk path’ making the work potentially relevant when we explored Reinforcement Learning approaches.  Eventually we moved away from these methods as our actor ‘choices’ i.e. picking a post had very little effect on the State/Environment hence we reject RL methods, despite a thorough investigation of use cases and an investigation of relevant works such as those in [7] to [10] below. The RL approach is effectively too contrived for our objective.
        ''')
    related_work.info('''
        **Title:** Deep reinforcement learning with a combinatorial action space for predicting popular reddit threads.[7]  
        **Topic:** Reinforcement Learning on Reddit data    
        **Implication for our project:** Similar domain space, different approaches.  Showed how a simulator might be used to reconstruct ‘Trees’ to set up and test sequential decision making. Related to our main task but not identical.
        ''')
    related_work.info('''
        **Title:** Deep reinforcement learning with a natural language action space.[8]  
        **Topic:** Reinforcement Learning for NLP - Text based games    
        **Implication for our project:** Illustrated the large action space issue for deep Q-learning in NLP.  Helps understand why the Tree approach was taken in [8] in order to re-use approach from text based games when seeking to predict karma on reddit.  
        ''')
    related_work.info('''
        **Title:** Deep reinforcement learning for NLP.[9]  
        **Topic:** A primer on Reinforcement Learning for NLP    
        **Implication for our project:** Simple introduction and overview to papers in the domain.  
        ''')
    related_work.info('''
        **Title:** SocialSift: Target Query Discovery on Online Social Media With Deep Reinforcement Learning.[10]  
        **Topic:** Generation of SQL queries using Reinforcement Learning    
        **Implication for our project:**  Sets out an online method (via API) of testing created text (‘Queries’) where the returned results are classified to create a reward for the RL policy Pi. The text of the query is effectively keywords that are extracted from the corpus of the previous query history and returned results.
        ''')
    related_work.info('''
        **Title:** Real-Time Predicting Bursting Hashtags on Twitter.[11]  
        **Topic:** Predicting hashtag bursts on Reddit    
        **Implication for our project:**  Similar data and has a temporal aspect to the prediction challenge. The definition of a ‘burst’ in this paper used a maximum function of hashtag counts over a 24 hour period to define a burst. Our early exploration of popularity as defined in Augury took inspiration from this, but was later adapted to our 3 hour target which better suited our objective. Augury also took some inspiration from the classification approach used in this paper.
        ''')

st.subheader("") #create blank space
st.subheader("Ethical Considerations")
st.write('''
    There are clearly ethical implications flowing from broadcasting messages on social media and the related investments to which these messages refer, as highlighted by the ‘Reddit Rally’ described above. In project Augury we are only looking at the popularity of posts, and we have not correlated this to market activity, which could be an extension of this work. To some extent this research has already been investigated by Muxi Xu  in 2021 [12] and also by Hu et al [13]. Therefore we have made no further mitigations in our project related to market ethics.  
    
    Social Media is often thought of as an open and public forum of discussion.  But an important ethical consideration of any data science project related to social media is that saying something in “public” may not necessarily mean “consent” to using a person’s name or username in published research [14].  While we did not have any need for usernames in the analysis we were performing, we did gather usernames into our database in order to track certain features of each Reddit post, and we are aware that if we were to ever use those usernames in future work, our database would could provide us with an already anonymized id number to protect each individuals’ privacy.  
    
    A different aspect of  the free and open nature of social media, and Reddit in particular, is that it can  lead to environments which become toxic in a number of ways.  To ameliorate this we did take some action in our post-processing of our database.  The subreddits which are looking at Investment are typically more professional in nature so the main way in which toxicity occurs in these fora is through the use of profane language. Thus, in our pipeline we have removed this language from our analysis using the python module profanity-filter which replaces profane words with the character “*”  [(docs)](https://pypi.org/project/profanity-filter/).  

    ''')


st.subheader("") #create blank space
st.header("Our Project Workflow")
st.write("The below graphic illustrates our project Augury workflow.  Below we will provide more details on each component of the workflow.")
project_pipeline_image = Image.open('blog_assets/project_pipeline3.png')
col1, col2, col3 = st.columns([1,5,1]) #column trick to center on the webpage
with col1:
    st.write("")
with col2:
    st.image(project_pipeline_image, caption='Project Augury Workflow')
with col3:
    st.write("")


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
st.subheader("Feature Engineering")
feature_table = st.container()
with feature_table:
    st.write("Based on our review of Related Works, our intuition, our understanding of Reddit, and some of our initial Exploratory Data Analysis (described in the next section), we chose to engineer the following features for use in our prediction task.  (*Click on a feature for description & rationale*)")
    with feature_table.expander("Number of comments per hour"):
        st.markdown('''
            *Description:*  This is a count of the comments each post has received, divided by the number of hours that have elapsed since the post was created.  

            *Rationale:*  Our research and intuition told us that the number of people commenting on a post is an indicator of likely popularity.
        ''')
    with feature_table.expander("Author Karma for the Post"):
        st.markdown('''
            *Description:*  We tracked the karma of both comment and post authors at the time of making either a post or a comment.  Reddit karma is a score a user receives based on the user's activity.  

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
            *Description:*  We used the SBERT library to encode the text of both Posts and Comments.  The SBERT interface was simple to use and produced in effect 380+ features for our classifiers for each post.  It provides a vectorial embedding of a sentence.  For the posts, this provides a vectorial space representation of each sentence that can be compared.  [(docs)](https://www.sbert.net/docs/pretrained_models.html)     

            *Rationale:*  BERT produces state of the art word embeddings for many NLP tasks.  However, it is computationally intensive to ‘fine tune’ for specific tasks like classification and this will be especially slow for sentence similarity tasks.  For this reason we selected SBERT[15] which uses the BERT weights as a base model and creates fixed length embeddings for more efficient sentence comparison tasks.    
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
            *Description:*  We used the SBERT library to encode the text of of the top 5 comments for a post (if available), just as we did for Posts.  It provides a vectorial embedding of a sentence.  For the comments we created a centroid vector of all the comments by averaging each comments’ SBERT embedding.  [(docs)](https://www.sbert.net/docs/pretrained_models.html)       

            *Rationale:*  BERT produces state of the art word embeddings for many NLP tasks.  However, it is computationally intensive to ‘fine tune’ for specific tasks like classification and this will be especially slow for sentence similarity tasks.  For this reason we selected SBERT[15] which uses the BERT weights as a base model and creates fixed length embeddings for more efficient sentence comparison tasks.  
        ''')

st.subheader("") #create blank space
st.subheader("Exploratory Data Analysis (EDA)")
st.write('''
    As mentioned above, we were successfully scraping Reddit consistently starting on March 1st of 2022.  We began performing EDA with the data we had gathered so far on March 28th of 2022, which represented roughly 1,200 posts tracked over a 24 hour period.  
    
    _Exploring our target y-variable for our use case:_  

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
    _Exploring basic feature data about the posts:_  

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
    
    Given the global nature of online communities such as Reddit and the fact that the data show was scraped and stored in UTC time, we don't want to read much into these relationships, but we do notice some interesting differences in popularity based on when the post is created.  For instance, in our sample we see a higher median maximum popularity for posts created on the weekend (Friday, Saturday, Sunday).  This makes some intuitive sense.  In regard to what hour of the day the post was created, we would not say there is any strong or consistent trend within our sample about what hour the post is created, though in future research it could be interesting to dig into some of the spikes we see in the chart.  

    ''')
temporal_image = Image.open('blog_assets/eda_temporal_chart.png')
st.image(temporal_image, caption='Analysis the median maximum popularity achieved based on the day or time a post is created')
st.write('''
    _Exploring initial feature importance with a basic classification model:_  

    Using the conclusion derived at the beginning of this section, that our project would focus on predicting “popularity” within 3 hours, we were able to do some initial exploration of how both the basic features and more advanced Natural Language Processing (NLP) features might influence our eventual model.  In order to do this, we used a basic “plain vanilla” Logistic Regression (LR) model on our sample data set, using the set of engineered features we had been considering. The below chart summarizes the feature importances of that initial look:  
     - NLP-based SBERT based features look by far the most important.  
     - We are surprised to see the other NLP-based features, measuring Sentiment of the text, as almost irrelevant.  We will be interested to see if this same trend persists after our hyperparameter tuning process.  
     - The other basic features and the one-hot-encoded hours of the day (x0_0 … x0_23) and days of the week (x1_0 … x1_6) also appear almost irrelevant.  

    This initial look at a basic model gives us a strong indication from this work that NLP-based SBERT features will be very important to our project.

    ''')
eda_feature_importance_image = Image.open('blog_assets/lr_feat_imp_short.png')
st.image(eda_feature_importance_image, caption='EDA Feature Importance in a plain vanilla Logistic Regression')
st.write('''
    In summary, the above EDA was very helpful in determining our classification of "popular" versus "not popular", and also gave us some initial expectations about how relevant our features may be.
    ''')
    

st.subheader("") #create blank space
st.header("Modeling Inferences & Evaluation")
st.subheader("Model Candidates")
st.markdown('''
	We selected three Supervised Learning models on which to attempt our Reddit post popularity prediction.  We kept our model exploration within the popular Scikit-Learn python library for the reason of more efficient and consistent modeling pipeline management.
	 - Logistic Regression (LR):  LR was chosen as it also appeared in some of the related work that we reviewed, and this simple classifier is often used as a “baseline” prediction model.  In this way, Logistic Regression is used for classification tasks in the same way Linear Regression is used for regression tasks.  [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
	 - Support Vector Classification (SVC): SVC was chosen as it also appeared in the related papers that were pursuing similar goals to our Augury project.  This makes sense, as Support Vector Machines are very effective in a high dimensional space, this type of project makes use of NLP-based features making a vectorial space representation of text data (in our case, Reddit posts and comments).  [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)  
	 - Gradient Boosting Classifier (GBDT):  GBDT was chosen based on our intuition and also that Gradient Boosted classifiers are popular in business and do well in data science competitions.  [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
	''')


st.subheader("Hyperparameter Tuning Pipeline")
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
    We have been successfully scraping from Reddit since March 1st of 2022, and we are going to be training and validating our model choices on data up until April 18th of 2022, or 48 days of hourly scraping.  Therefore, our "unseen" test set of data will be from April 18th to April 23rd of 2022, or 5 days of hourly scraping.  
    
    _Here is how these splits translate into the percentages of our overall data and number of individual posts:_  

    ''')
col1, col2, col3 = st.columns(3)
col1.info("**Training Data**")
col2.info("**Validation Data**")
col3.info("**Testing Data**")
col1.write("71% of data")
col2.write("18% of data")
col3.write("11% of data")
col1.write("2,140 posts")
col2.write("535 posts")
col3.write("346 posts")
col1.write("~17,000 underlying comments")
col2.write("~3,000 underlying comments")
col3.write("~2,000 underlying comments")


st.write("") #blank space
st.write('''
    **Tuning Pipeline:**  
    Our full modeling pipeline can be reviewed in our GitHub repository linked at the top of this page, and below we will be describing many of its components, but there are a couple important comments to make the creation of this process.  
     - We felt it was important to ensure we didn’t accidentally create “data leakage” in our results, and so as a part of our pipeline we use tools that would not allow Validation Data to influence some of our feature transformations.  
     - We found it useful to break up our pipeline iterations by model (sometimes by different sub-type of model) and save the results.  This made it easier to better understand compute time and find where the “pain points” are in the process.  It is important to test these large processes in steps before launching the whole thing as small mistakes can really have a large impact and require full recalculation.  
    The successful creation of our hyperparameter tuning pipeline allowed us to create the many different model options we review in the next section.  
    
    ''')
st.write('''
    **Tuning Iterations:**  
    We iterated through over 1,000 combinations of parameters across LR, SVC, and GBDT models, for which we needed to develop both objective and subjective methods of making parameter choices.  
    ''')
st.write('''
    _LR Parameter Options:_ For LR we flipped between the solver parameter, the penalty type, and a range of values of C we've seen before in our course work.  
    
    _Parameter dictionary used in our code_
    ''')
st.code('''
    parameters = [{"clf__C":np.logspace(-5, 10, num=16, base=2),
                "clf__solver":["liblinear", "lbfgs"],
                "clf__penalty":["l1", "l2"],
                }]
    ''', language="python")
st.write("") #blank space
st.write('''
    _SVC Parameter Options:_ We looked at each of the four kernels individually to avoid errors that would make us lose a lot of compute time. The range for C and gamma was taken from A Practical Guide to Support Vector Classification by Chih-Wei Hsu et al.[16], these were slightly more thorough in their tests than other information available.  
    
    _Parameter dictionary used in our code (approximately)_
    ''')
st.code('''
    parameters = {"clf__C":np.logspace(-5, 15, num=21, base=2), 
                "clf__kernel":["rbf","linear","poly","sigmoid"],
                "clf__degree": [3, 4, 5], #only related to the polynomial kernel
                'clf__gamma': np.logspace(-15, 3, num=19, base=2)}
    ''', language="python")
st.write("") #blank space
st.write('''
    _GBDT Parameter Options:_ We tuned the GradiantBoostingClassifier on the following five parameters. These were picked from great recommendations from MachineLearningMastery [17] as well as some adjustments for speed (reducing the number of total folds) and  introducing subsamples for Stochastic Gradient Boosting as per the recommendation of the Friedman 1999/2002 paper[18] that is explained in the Sci-Kit Learn documentation.  
    
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
    _Other notable parameter decisions prior to the tuning iterations:_ The following decisions were needed for reasons other than finding the best performing model, which we are sharing along with our rationale in case the reader finds it instructive in their own projects.  
     - For all three GridSearchCV pipelines we set refit="F1" because we were passing a scoring directory, and F1 is our most important score.  
     - In both LR and SVC models, we set class_weight="balanced". Since our classes are imbalanced, this options ensures it will adjust the loss function by weighting the loss of each sample by its class weight (making the smaller class weigh more).       
     - In both LR and SVC models, we set max_iter=100000.  Since our vectorial space was very large (caused by NLP-based features) the default value of 100 did not converge.
     - In the LR model, we set multi_class="ovr" because our problem was binary (0 or 1 classification) and not multinomial.
     - In the GBDT model, we reviewed and decided to keep the default parameters min_samples_split=2 and min_samples_leaf=1 since our dataset is not large enough to warrant bigger splits.  
    ''')
    
st.write("") #blank space
st.write(''' 
    **Hyperparameter Decision Process:**  
    With over 1,000 different options produced from our hyperparameter tuning process, we used both objective and subjective methods of coming to a decision about which hyperparameters were most appropriate, and thus which of the three styles of model to ultimate use for our prediction task.  
    
    For each iteration, we output the F1 Score calculation for each model, both on the training data and validation data.  F1 Score felt most appropriate as it has a better ability to account for imbalanced classification data such as ours, where only a small percentage of posts would become "popular.  Also, the F1 Score was the metric (and sometimes sole metric) used in similar projects we saw in related works.  Using this information, we went through these steps in each notebook:
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
    **Chosen Hyperparameters:**  
    The above process resulted in the following hyperparameter decisions:
     > **LR**:  
     >> Solver: lbfgs   
     >> Penalty: L1  
     >> C: 0.0125 
     > _Rationale:_ Our LR tuning charts give a very clear decision on the 'penalty' parameter and 'solver' parameter, as those options have the highest F1 scores on Validation Data.  However, the choice of C is more nuanced.  While the chart shows a linear trend of a higher F1 score for lower values of C, we stated above a subjective preference that we did not want to choose either minimum or maximum values of our parameter choices, therefore we settle on a 'C' value of 0.125 near the minimum.
     >   
     > **SVC**:  
     >> Kernel: rbf    
     >> C: 0.125  
     >> Gamma: 0.0078125  
     > _Rationale:_ Our SVC tuning charts gives a clear preference for the "rbf" kernel based on median F1 score on Validation Data, but the choice of the other two parameters is more nuanced.  For 'C', there seems to be a non-perfect linear relationship where lower values result in higher F1 scores on Validation Data, so again we choose a value of 0.125 which is close to, but not at, the minimum 'C' value (due to our subjective guidelines above).  For 'gamma', we see a drop-off in F1 score above values of 0.03, so we subjectively chose 0.0078125, which is close to that value.
     >  
     > **GBDT**:  
     >> Learning_rate: 0.15  
     >> N_estimators: 150  
     >> Max_depth: 3  
     >> Max_features: sqrt  
     >> Subsample:  0.5  
     > _Rationale:_ Our GBDT tuning charts give a clear preference for 'sqrt' for max_features and 3 for max_depth.  For the subsample parameter we subjectively chose 0.5 as the other options had Training Data scores that felt "too good".  For n_estimators, the chart exhibits slight curve where we could pick a "peak" F1 score on Validation Data at 150 estimators.  For learning_rate, while there is a linear relationship between F1 and parameter value, due to our "subjective" guidelines above we choose 0.15 which is near, but not at, the maximum value. 

    ''')




st.write("") #blank space
st.subheader("Model Choice")
st.write('''
    Using the hyperparameter choices above, we looked at the cross-validation performance results for each of the three "tuned" model options.  
    
    We are primarily looking at F1 Score results for our Validation Data, though we did maintain an awareness of both F1 and Accuracy across both the training and validation data sets. In the table below, we see that the model that had the highest F1 Score on the Validation data is our tuned SVC model, which was slightly ahead of the LR option.  The GBDT option had the worst F1 score on Validation Data, and also had suspiciously high values on Training Data, so we decided not to pursue GBDT any further.  
    
    Therefore, we choose the SVC model to be the "Project Augury" model, and we will use the more simple LR model as a "baseline" comparison for our Test Data.  

     
    ''')
hpt_f1_df = pd.DataFrame(data={
    'Tuned_Model': ['LR','SVC','GBDT'],
    'F1_Score_Validation': [0.5168,0.5230,0.3509],
    'F1_Score_Training': [0.5901,0.7271,0.9411],
    })
hpt_accuracy_df = pd.DataFrame(data={
    'Tuned_Model': ['LR','SVC','GBDT'],
    'Accuracy_Validation': [0.7350,0.7271,0.7948],
    'Accuracy_Training': [0.7754,0.7531,0.9769],
    })

st.info("**Model Results on Validation Data:**")
col1, col2, col3, col4 = st.columns([1,1,1,1])
col1.info("**Score**")
col2.info("**LR**")
col3.info("**SVC**")
col4.info("**GBDT**")
col1.info("F1")
col2.warning("0.5168")
col3.success("0.5230")
col4.error("0.3509")

with st.expander("Click here to see the full table of F1 and Accuracy for both Validation and Training Data:"):
    st.table(hpt_f1_df)
    st.table(hpt_accuracy_df)

st.write("") #blank space
st.subheader("Model Feature Importance")
st.write('''
     The chart below illustrates the feature importances for our chosen SVC model:  
      - The SBERT-based features for both posts and comments have the strongest importance in the model, with the encoding of Posts being stronger than that of the underlying comments. This strong importance fits our original intuition and rationale for using SBERT that we described in the Features section above, as well as the initial “basic” model importance that we performed in our EDA.  
      - The upvotes that comments receive (avg_comment_upvotes_vs_hrs) and overall number of comments a post receives (number_comments_vs_hrs) are the third and fourth most important features, respectively. This matches our expectation based on the high correlation to popularity we found in our EDA.
      - The two temporal features of our model related to the hour and the weekday a post is created (time_hour and day_of_week) show some importance but not much, which is consistent with our intuition and our EDA findings.  
      - The two features related to a post or comment author's karma also show some, but weak, importance. This is consistent with the weak correlations we saw in our EDA.
      - The _biggest surprise_ of both our model’s feature importance and the prior analysis in our EDA is that the sentiment of both the post and the underlying comments has almost no impact in our model.  This is very different from our intuition described in the Features section above that text with a very positive or very negative sentiment might drive the activity and popularity of an individual post.  

    ''')
#feature_importance_image = Image.open('blog_assets/SVC_feature_importance_v1.png')
#feature_importance_image = Image.open('blog_assets/SVC_feature_importance_v2.png')
#feature_importance_image = Image.open('blog_assets/SVC_feature_importance_v3.jpg')
#feature_importance_image = Image.open('blog_assets/SVC_feature_importance_v4.png')
feature_importance_image = Image.open('blog_assets/SVC_feature_importance_v5.png')
st.image(feature_importance_image, caption='Aggregated Feature Importance in our SVC Model')
st.write('''
    _Challenges to getting feature importance:_  We discovered two key challenges to creating this feature importance summary for our project, which might be instructive to the reader in their own work.  The first relates to feature choices and the second relates to model choices.  
     1. Related to feature choices, four of our features summarize above are actually an aggregate importance of several items.  These are the SBERT and temporal features.  This is because in feature engineering, the temporal features required one-hot encoding (for instance, the 7 days of the week become 7 new columns) and the SBERT features generate many columns to represent the sentence encodings (see the above Features section for more details).  So, in looking at feature importances, we had to create additional methods beyond what was in the SciKit Learn libraries in order to aggregate the importances.  The key take-away for the reader is that the choice of features can require additional steps later in the project to ensure interpretability of the model.  
     2. Related to model choices, the SVC model we chose did not have a straightforward 'model.feature_importance_' method like GBDT in SciKit Learn. Instead we had to run a very computationally expensive process called ‘get feature permutations’, essentially this runs a type of cross validation process by shuffling the features to see their relative importance to the model. It produces a ‘bunch object’ from which the importances for each feature can be derived as shown in the chart above.  The key take-away for the reader is that model choices can lead to different 'forks' in your pipeline of work to analyze your results.  

    ''')


st.write("") #blank space
st.subheader("Model Performance")
st.write('''
    With our tuned SVC model and our "baseline" LR model chosen in the prior section, we measure the ultimate success of these models on previously unseen Test Data that was set aside when we began our hyperparameter tuning process (_see the "Project train / validation / test split" section above for details_).  
    
    The below table illustrates that on our primary scoring metric, F1, our SVC model performs better than the baseline model.  We can also see in the expanded table that our SVC model also performs better than the baseline on Accuracy.  Therefore, we continue to use our tuned SVC model in the next section where we put this model into a production environment before concluding our project.  
     
    ''')
test_f1_df = pd.DataFrame(data={
    'Model': ['LR','SVC'],
    'F1_Score_Test' : [0.4577,0.500],
    'F1_Score_Validation': [0.5168,0.5230],
    'F1_Score_Training': [0.5901,0.7271],
    })
test_accuracy_df = pd.DataFrame(data={
    'Model': ['LR','SVC'],
    'Accuracy_Test': [0.6850,0.7110],
    'Accuracy_Validation': [0.7350,0.7271],
    'Accuracy_Training': [0.7754,0.7531],
    })
st.info("**Model Results on Test Data:**")
col1, col2, col3 = st.columns([1,1,1])
col1.info("**Score**")
col2.info("**LR** (baseline)")
col3.info("**SVC** (Augury)")
col1.info("F1")
col2.error("0.4577")
col3.success("0.500")

with st.expander("Click here to see the full table of F1 and Accuracy on Test, Validation, and Training Data:"):
    st.table(test_f1_df)
    st.table(test_accuracy_df)

st.write("") #blank space
st.subheader("Real-Time Model Prediction")
#load pkl'd classifier (clf)
filename = "models/SVC_rbf_final_model.pkl" 
#filename = "models/LogisticRegression_final_baseline_model.pkl" 

### cached model for blog efficiency
@st.cache
def load_model():
	return pickle.load(open(filename, 'rb'))

clf = load_model()
#clf = pickle.load(open(filename, 'rb'))

st.markdown('''
    We wanted to give the readers of this blog the opportunity to see our model's predictions on the current data in Reddit!  Please take a look at the instructions below, select your options, and take a look at our model's prediction!  
     - The first selection to make is what subreddits you would like to scrape from.  Because our model was generalized around only a single theme of "investing", we will limit you to the four subreddits we used in our study described above.  This scraping will be "live" directly to Reddit using similar code to that we built in AWS to scrape our project data.  
     - The second selection is the number of the newest posts you want to try to scrape for each subreddit.  Note that we will only scrape posts created within the last hour, which mirrors the process of our project's scraping process in AWS.  So, readers should be aware that you may see less posts than you are attempting to scrape unless it is a particularly active day for the subreddit.  
     - The posts that are available will have all the same data collected that we use for our featurization process in our project.  After we engineer the features using the same pipeline as our project, we will use our trained Classifier mode to make a prediction on each post.  
     - The output of this "live" prediction process is a table showing some information about your post, as well as the probability of a post becoming "popular" according to our model.  The higher the probability, the greater the chance our model thinks that post will become "hot" on the subreddit.  

    ''')

### credentials hidden by Streamlit for Reddit
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
if len(subreddit_selection_list)==0:
    st.error("You have not selected any subreddits!")

for item in subreddit_selection_list:
    subreddit_scrape_list.append(item[2:]) #removes the r/
subreddit_scrape_list = subreddit_scrape_list
#st.write("You selected:", subreddit_scrape_list)

#n_posts = 5
n_posts = st.selectbox("Select the number of posts you'd like to scrape from each subreddit:", (1,3,5))
n_posts = int(n_posts)

if (len(subreddit_selection_list)==1) & (n_posts==1):
    st.warning("We recommend you choose more than 1 post if you have chosen only 1 subreddit!")
if (len(subreddit_selection_list)==2) & (n_posts==1):
    st.warning("We recommend you choose more than 1 post if you have chosen only 2 subreddits!")


n_comments = 5 
hrs_to_track = 1 #number of hours to track a post/submission
#time_of_batch = datetime.utcnow().replace(microsecond=0)                                      
char_limit = 256 #character limit for the text fields in the database


st.write("") #blank space
if st.button("Predict Reddit Popularity!"):
    st.write("Number of posts we'll try to collect:", n_posts*len(subreddit_scrape_list))

    time_of_batch = datetime.utcnow().replace(microsecond=0)
    try:
        st.write("Gathering new posts from Reddit...")
        new_submission_list = Team_Augury_blog_praw_functions.blog_submission_list(reddit=reddit, time_of_batch=time_of_batch, hrs_to_track=hrs_to_track, n_posts=n_posts, subreddit_scrape_list=subreddit_scrape_list)
        #st.write("len(new_submission_list)",len(new_submission_list))
        if len(new_submission_list)==0:
            st.warning("There are no new posts within the last hour for your selection.")
        else:
            st.write("Number of new posts we've identified on Reddit:", len(new_submission_list))
    except:
        st.error("A problem occurred contacting Reddit.")

    try:    
        st.write("Starting feature engineering...")
        post_data, comments_data = Team_Augury_blog_praw_functions.blog_scrape_dataframes(reddit=reddit, time_of_batch=time_of_batch, n_comments=n_comments, char_limit=char_limit, new_submission_list=new_submission_list)
        #st.table(post_data)
        #st.table(comments_data)
        feature_df = Team_Augury_blog_praw_functions.blog_feature_creation(post_data, comments_data)
        #st.table(feature_df)
        output_df = feature_df[['sr','post_id','post_text']].copy() #need this to attach predict_proba to later...
        #st.write("output_df",output_df)
        del post_data, comments_data
        st.write("Number of posts for which we engineered features:",len(output_df))
    except: 
        st.error("A problem occurred when creating the features.  We suggest either modifying your selections or waiting a few minutes to try again.")

    try:
        st.write("Starting transformation of features into our model's format...")
        feature_df = Team_Augury_blog_praw_functions.blog_X_values(feature_df)
        #st.write("X_values for pkl'd model")
        #st.table(feature_df)
        #st.write("len(feature_df.columns):",len(feature_df.columns))
    except:
        st.error("A problem occurred when transforming the features into the model's desired format.")

    try: 
        st.write("Starting prediction process with our model...")
        #predictions = clf.predict(feature_df)
        #st.write("predictions...", predictions)
        prediction_probas = clf.predict_proba(feature_df)
        #st.write("prediction probabilities...", prediction_probas)
        del feature_df #no longer need feature_df
    except:
        st.error("A problem occurred in making the model predictions.")
            
    try: 
        st.write("Starting to create the model output...")
        output_df = pd.DataFrame({
            'Subreddit': output_df['sr'],
            'Post ID':  output_df['post_id'],
            'Post Title':  output_df['post_text'],
            'Popular Probability':  pd.Series(prediction_probas[:, 1].round(2)),
            })
        del prediction_probas #no longer need this item
        st.write("**Post Popularity Prediction** (_Sorted most likely to least likely_)", 
                output_df.sort_values(['Popular Probability'], ascending=False).reset_index(drop=True))
        st.write("Check out the subreddits you selected to see if you can find the post(s) you just predicted.  You can do this by clicking on the 'New' icon in the header (illustrated below).  A high probability means that our model thinks 'new' post should be 'popular' in 3 hours and be listed towards the top of the subreddit for its 'hot' category.  ")
        reddit_image = Image.open('blog_assets/reddit_header.png')
        st.image(reddit_image, caption='subreddit header')
        st.caption("Content Warning:  The below links will take you to Reddit's website, where there may be content that might be offensive.")
        st.write(''' 
             - [r/investing](https://www.reddit.com/r/investing/)  
             - [r/wallstreetbets](https://www.reddit.com/r/wallstreetbets/)  
             - [r/StockMarket](https://www.reddit.com/r/StockMarket/)  
             - [r/stocks](https://www.reddit.com/r/stocks/)  
            ''')
    except:
        st.error("A problem occurred in making the output dataframe.")
    
st.caption("Content Warning:  While we apply a profanity filter to both the post and comment text, the output of this prediction process will show the text of individual Reddit posts and could show content that may be harmful to some readers.")
    

    

st.write("") #blank space
st.header("Conclusions & Future Work")
st.write('''
    We were able to define a temporal measure of Reddit post popularity when considering similarly themed subreddits, with an aim of predicting what will become popular within three hours of being posted.  Our tuned SVC model attained an F1 classification performance higher than that of a baseline LR model, and also impressive compared to what we have seen in related work.  In creating this model, we witnessed very strong importance of the NLP-based SBERT encodings of text, both the text of the posts and of the underlying comments.  
 
    Given more time we would like to continue the work, growing our data set in the same way our pipeline was designed for a longer period of time.  We acknowledge that our data set was smaller than we initially hoped for, related to technical difficulties, so it would be only natural to want to revisit our current analysis and possibly expand on some areas of  curiosity that came up during our project:  
     - Test our trained classifier with more data captured over a longer period of time.  
     - Run our same pipeline with more data to see if we reach a different model choice, possibly using RandomizedSearchCV instead of GridSearchCV as our choice of cross-validation.  The former uses modules that could make our pipeline work with more data much faster.  
     - Set a higher threshold for achieving “popularity” for a post, and see if we can still generate classification performance that is as impressive.  
     - Further exploration of existing features.  For instance, our feature importances work gave a surprising result in that VADER sentiment was relatively unimportant, we would like to explore this and the inter-dependencies between features.  Also, it would be interesting to explore more of the small patterns we saw in our EDA of the weekday or hour a post was created.  
     - Further experimentation with new features.  For instance, we began to explore the networks that are created among Reddit users for both posts and comments and think this is worthy of further work for example.  
     - Explore methods of creating a more “balanced” data set (between popular / unpopular) in order to improve the accuracy of predicting popularity.  
 
    There is potential to use the database and process we have developed for other projects.  Ideas that we entertained either at the start of this project or as we progressed through it include:  
     - Financial market / stock prediction. We have observed related work in that domain that, on face value, seemed overly optimistic in its ability to predict trends. For our work to be developed in that direction we may consider scraping additional items from Reddit about the posts or comments, but that is not an insurmountable hurdle.  
     - “Influencer” analysis.  Using the information we have about the Reddit users, we could conduct a network analysis study to identify the most influential users in the “investing” themed subreddits and see if their influence is limited to just one or more subreddits.  However, for this kind of work we would naturally consider the ethical pitfalls of working with user data with a goal of identification, and take necessary ethical precautions to prevent harm.  
      
    >That concludes the Team Augury project to predict what Reddit posts about investing are likely to become popular.  We thank you for reading!  
    > --- Antoine, Chris, & Erik

    ''')

st.write("") #blank space
st.write("") #blank space
st.write("") #blank space
st.markdown('''---''')
st.header("Appendix: Statement of work")
st.write('''
The team worked across the entire project, but the below highlights the areas each team member either focussed, led or made a major contribution:  

Antoine Wermenlinger led on the Supervised Learning aspect of the project from the long list of models, through to the down select and hyperparameter tuning. He also discovered the SBERT encoding and python module as appropriate for this task and ran the initial experiments on this. Antoine initiated the feature engineering work and was the originator of the project's core goal -- to predict popularity of Reddit posts.  

Chris Lynch led on the instantiation of our AWS instance, account and functions, notably the Lambda Function, Event Bridge Scheduling and the creation of our RDS (a postgreSQL database). In the early phases of the project Chris led on the Reinforcement Learning aspects of the project creating a Reddit Simulator environment and a Q learning model, which eventually we decided not to use. He worked on explainability of our models by exploring feature/permutation importances. Chris initiated the draft for our team Blog post.  

Erik Lang led on the Social Media scraping and analytics aspects of the project, creating the PRAW based code that scraped and cleaned the raw Reddit data that was written into our postgreSQL instance. He made major contributions to feature engineering, EDA, visualizations and team standups. Finally Erik created our Streamlit instance, that is the host site for this Blog. 
    ''')


st.write("") #blank space
st.header("Appendix: References")
st.write('''
[1] Š. Lyócsa, E. Baumöhl, and T. Vyrost, “YOLO trading: Riding with the herd during the GameStop episode,” ˆ EconStor, 2021. [Online]. Available: https://www.econstor.eu/handle/10419/230679  [Accessed: 06-Apr-2022].  
[2] Kim, Juno. "Predicting the Popularity of Reddit Posts with AI." arXiv preprint arXiv:2106.07380 (2021).  
[3] S. Deaton, S. Hutchison, and S. J. Matthews, “Using Machine Learning to Predict the Popularity of Reddit Comments,” seandeaton.com, 2017. [Online]. Available: https://seandeaton.com/publications/reddit-paper.pdf  [Accessed: 06-Apr-2021]  
[4] Rohlin, Tracy M. Popularity prediction of Reddit texts. San José State University, 2016.  
[5] A. Shuaibi. Learning, General Machine. "Predicting the Popularity of Reddit Posts." Spring 2019  
[6] Li, Cheng, et al. "Deepcas: An end-to-end predictor of information cascades." Proceedings of the 26th international conference on World Wide Web. 2017.  
[7] Ji He, Mari Ostendorf, Xiaodong He, Jianshu Chen, Jianfeng Gao, Lihong Li, and Li Deng. 2016b. Deep reinforcement learning with a combinatorial action space for predicting popular reddit threads. EMNLP.  
[8] Ji He, Jianshu Chen, Xiaodong He, Jianfeng Gao, Li-hong Li, Li Deng, and Mari Ostendorf. 2016a. Deep reinforcement learning with a natural language action space. ACL.  
[9] Wang, William Yang, Jiwei Li, and Xiaodong He. "Deep reinforcement learning for NLP." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts. 2018.  
[10] Wang, Changyu, et al. "SocialSift: Target Query Discovery on Online Social Media With Deep Reinforcement Learning." IEEE Transactions on Neural Networks and Learning Systems (2021).  
[11] Kong, Shoubin, et al. "Real-time predicting bursting hashtags on twitter." International Conference on Web-Age Information Management. Springer, Cham, 2014.  
[12] Xu M. NLP for Stock Market Prediction with Reddit Data 2021. Stanford.  
[13] Hu, Kevin, Daniella Grimberg, and Eziz Durdyev. "Twitter Sentiment Analysis for Predicting Stock Price Movements."  
[14] Zimmer, Michael T.  “OkCupid Study Reveals the Perils of Big-Data Science”, Wired, 2020.Available:https://www.wired.com/2016/05/okcupid-study-reveals-perils-big-data-science  [Accessed: 15-Apr-2022].  
[15] Reimers, Nils, and Iryna Gurevych. "Sentence-bert: Sentence embeddings using siamese bert-networks." arXiv preprint arXiv:1908.10084 (2019).  
[16] Chih-Wei Hsu, et al. A Practical Guide to Support Vector Classification. 2016.  National Taiwan University.  
[17] Brownlee, Jason.  “How to Configure the Gradient Boosting Algorithm”, Machine Learning Mastery, 2016. Available: https://machinelearningmastery.com/configure-gradient-boosting-algorithm/  [Accessed: 7-Apr-2022]  
[18] Friedman, Jerome H. “Stochastic Gradient Boosting”, Computational Statistics & Data Analysis, 2002. Available: https://www.researchgate.net/publication/222573328_Stochastic_Gradient_Boosting [Accessed: 20-Apr-2022]  
    ''')


### End
