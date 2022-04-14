import streamlit as st
import pandas as pd
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
st.write('''
In order to request data from Reddit’s API, we decided to use the widely known PRAW python library to do our data scraping.  Our approach to scraping was informed primarily by studying the PRAW online documentation [15].    

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
st.image(db_schema_image, caption='Augury Database Schema in AWS')

st.subheader("") #create blank space
st.subheader("Exploratory Data Analysis (EDA)")
st.write('''
    Our objective is to predict whether or not a post will become “popular”.  A “popular” post is one that would be near the top of the subreddit when a user goes to the webpage or opens the app (in Reddit’s vocabulary, the post is “Hot”).  From perusing Reddit’s own message boards, we understand that a rough approximation for this measure of “popularity” is:
    ''')
st.latex(r"popularity = \frac{upvotes}{hours}")
st.write('''
    We illustrate our exploration of “popularity” in the below chart on a sample of our data, where the faint blue lines are the popularity of individual posts scraped hourly for 24 hours from March 1st to March 28th, 2022, or (roughly 1,200 posts).  Overlaid on this chart are three lines representing the 50th percentile (half the data), 80th percentile (the top quintile), and 90th percentile (the top decile).  
    ''')
popularity_image = Image.open('blog_assets/EDA_popularity.png')
st.image(popularity_image, caption='Proxy for Post Popularity')
st.write('''
    From this chart, we make two decisions about our prediction task.  First, that popularity can peak at different times, but using a post’s popularity at hour 3  seems to be appropriate for our prediction variable.  Second, we see a lot of the subsample achieves very low popularity, so we feel comfortable using a threshold for “popular” close to the top Quintile (the red line), or a “popularity” value of 10.  Thus, the determination of “popular” versus “not popular” in our prediction task is determined by if a post has a “popularity proxy” of over or under 10 by hour 3.  
    ''')
st.write('''
    Placeholder Text for more about EDA... maybe talk generally about factor correlations / network analysis?  We note in the features section below that we use certain features due to intuition or inclusion in Literature Review papers.
    ''')
    

st.subheader("") #create blank space
st.subheader("Feature Engineering (Option 1)")
feature_table = st.container()
with feature_table:
    st.write("Through a combination of our EDA, our intuition, and our understanding of Reddit, we chose to engineer the following features for use in our prediction task.  (*Click on a feature for description & rationale*)")
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
	 - Gradient Boosting Classifier (GBC):  GBC was chosen based on our intuition and also feedback from our professors that Gradient Boosted classifiers are often winning data science competitions lately.  [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
	''')


st.subheader("Hyperparameter Tuning")
st.write('''
    In order to perform hyperparameter tuning for these three models, we explored a few different options to perform cross-validation.  We ultimately decided on Scikit-Learn's GridSearchCV method [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).  
    We also explored RandomizedSearchCV and Nested Cross-Validation.  Since our dataset was relatively small the efficiency of RandomizedSearchCV was not required, and this also would have mad the Nested Cross-Validation difficult due to too small samples.  
    Importantly, we chose to use the standard cv object within GridsearchCV as it implements the stratified k-fold approach, thus ensuring we do not split without the much smaller “popular” class.  

    ''')
st.write("placeholder for image of Stratified K-Fold validation ??")
st.write("placeholder for discussing # of Train, # of Validate, # of Test (Unseen) data points!")
st.write('''
    **Tuning Iterations:**  
    We iterated through 1,062 combinations of parameters across LR, SVC, and GBC models.  
    ''')
st.write('''
    _LR Tuning:_ For LR we flipped between the _solver_ parameter and a range of values of C we've seen before in our course work.  
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
    _SVC Tuning:_ We looked at each of the four kernels individually to avoid errors that would make us lose a lot of compute time. The range for C and gamma was taken from A Practical Guide to Support Vector Classification (Chih-Wei Hsu et al. CITATION NEEDED BELOW), these were slightly more thorough in their tests than other information available.  
    _Parameter dictionary used in our code (approximately)_
    ''')
st.code('''
    parameters = {"clf__C":np.logspace(-5, 15, num=21, base=2), 
                "clf__kernel":["rbf","linear","poly","sigmoid"],
                "clf__degree": [3, 4, 5], #only related to the polynomial kernel
                'clf__gamma': np.logspace(-15, 3, num=19, base=2)}
    ''', language="python")
st.write('''
    _GBC Tuning:_ We tuned the GradiantBoostingClassifier on the following parameters. These were picked from great recommendations of MachineLearningMastery CITATION NEEDED BELOW as well as some adjustments for speed (reducing the number of total folds) and introducing subsamples for Stochastic Gradient Descent as per the recommendation of the Hastie et al. 2009 CITATION NEEDED BELOW paper explained in the scikit learn documentation.  
    _Parameter dictionary used in our code_
    ''')
st.code('''
    parameters = {"clf__learning_rate":[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                "clf__n_estimators":[50,75,100,125,150,175,200, 250],
                "clf__max_depth":[3,5,8],
                "clf__max_features":["log2","sqrt"],
                "clf__subsample":[0.5,0.75,1.0],}
    ''', language="python")

st.write(''' 
    **Hyperparameter Decision Process:**  
    With over 1,000 different options produced from our hyperparameter tuning process, we used both objective and subjective methods of coming to a decision about which hyperparameters were most appropriate, and thus which of the three styles of model to ultimate use for our prediction task.  
    
    For each iteration, we output the F1 Score and Accuracy calculation for each model, both on the training data and validation data.  Using this information, we went through these steps in each notebook:
     - Using three different Jupyter Notebooks to visualize the results the most appropriate tool for this process.
     - We began by viewing all of the F1 and Accuracy calculations for each model, as messy as it was, and noted any observable patterns.
     - We then created charts that looked at each hyperparameter, looking for levels of the hyperparameter that gave higher performance "all else equal".  By this we mean we took the median performance metric for each level of an individual hyperparameter.  From this method we were able to see general trends in the individual hyperparameters that resulted in higher or lower model performance.
     - Our _"objective"_ decision was to favor levels of a hyperparameter that yielded better scores, with a bias towards the F1 score.
     - Our _"subjective"_ decision was to favor levels of a hyperparameter that were neither at the extremes of our tuning ranges, nor resulted in results that looked "too good" and might indicate over-fitting.  
    
    ''')
st.write(''' 
    **Hyperparameter Decisions:**  
    The above process resulted in the following choices for hyperparameters:
     > **LR**:  
     >> Solver: liblinear   
     >> Penalty: L2  
     >> C: 0.25  
     > **SVC**:  
     >> Kernel: rbf  
     >> Penalty: L2  
     >> C: 4  
     >> Gamma: 0.00390625  
     > **GBC**:  
     >> Learning_rate: 0.15  
     >> N_estimators: 100  
     >> Max_depth: 3  
     >> Max_features: sqrt  
     >> Subsample:  0.5  

    ''')

st.write('''
    Performance metrics of the above "tuned" models:
    ''')
hpt_df = pd.DataFrame(data={
    'Tuned_Model': ['LR','SVC','GBC'],
    'F1_Score_Training': [0.75,0.80,0.93],
    'F1_Score_Validate': [0.39,0.49,0.39],
    'Accuracy_Training': [0.907,0.897,0.971],
    'Accuracy_Validate': [0.776,0.740,0.795],
    })
st.table(hpt_df)


st.subheader("Model Choice")

st.subheader("Feature Importance of the Model")

st.subheader("Model Performance (on unseen data)")

st.subheader("Real-Time Model Prediction")

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


