import streamlit as st
#from torch import hann_window
#import pandas as pd
#import joblib
#from sklearn.linear_model import LogisticRegression
from PIL import Image
import praw


# Title
st.title("Project Augury: Predicting which Investing posts on Reddit are likely to become popular")
st.caption(" **augury** _noun_; a sign of what will happen in the future     -- Cambridge Dictionary")
st.markdown(">Course: SIADS 697/698  \n>> Git Repository: link  \n>> Blog Post: link  \n>Authors:  \n>> Antoine Wermenlinger (awerm@umich.edu)  \n>> Chris Lynch (cdlynch@umich.edu)  \n>> Erik Lang (eriklang@umich.edu)")

st.header("Summary")
st.write('''
    Placeholder text
    ''')

st.header("Background")
st.subheader("Motivation")
st.write('''
    Reddit is a popular social media networking site. From an investing perspective there are a few subreddits that have discussions related to investing.  A subreddit is a forum on Reddit that is dedicated to a specific topic, in our case we have chosen four subreddits to investigate:

    Investing
    Wall Street Bets, aka WSB
    Stock Market
    Stocks

    WSB achieved some notoriety during the “GameStop” incident where Keith Gill, a formerly little known trader achieved gains of $40M plus in a short period of time referred to as the ‘Reddit Rally’. Reuters claimed his ‘punchy move’ sparked thousands of comments on the WSB subreddit, causing his post(s) to go viral and his stock position to dramatically rise in value.  

    Project Augury is focussed on exploring what makes a post on these subreddits to be popular. We are looking at a group of four subreddits as each subreddit itself has relatively low volumes of posts each day, when compared to the biggest subreddit’s like AskReddit, and our background research confirmed that predictive tasks on social media work better in thematic subreddits and do not necessarily generalize from one theme or subreddit to another.
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

st.header("") #create blank space
st.subheader("Related Work v2")
st.write('''
    placeholder text introducing the following tables of work we reviewed
    ''')
c = st.container()
col1, col2, col3 = c.columns(3)
col1.write("**Paper Title**")
col2.write("**Topic**")
col3.write("**Relevance to project**")
col1.write("[3] Reddit predictions")
col2.write("Supervised Learning approaches to predict popularity")
col3.write("This is a prediction task on similar data.")
c.write("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse ")
st.write("placeholder for text at the end of the literature review")
st.header("") #create blank space
st.write('''
    placeholder text introducing the following tables of work we reviewed
    ''')
c = st.container()
col1, col2, col3 = c.columns(3)
with c:
    col1.info("**Paper Title**")
    col2.info("**Topic**")
    col3.info("**Relevance to project**")

    col1.info("[3] Reddit predictions")
    col2.info("Supervised Learning approaches to predict popularity")
    col3.info("This is a prediction task on similar data.")
    c.info("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse) ")
    with c.expander("See implications for our project"):
        st.warning("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse ")





st.header("") #create blank space
st.subheader("Related Work v3")
st.write('''
    placeholder text introducing the following tables of work we reviewed
    ''')
c = st.container()
#col1, col2, col3 = c.columns(3)
with c:
    c.info("text")
    c.info("text")
    #c.write("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse) ")
    with c.markdown("**test**").expander('''
        Title: full text of paper title  \n
        Topic: text text  \n
        (Click for implications for our project)
        '''):
        st.write("Implications: The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse ")

st.write("placeholder for text at the end of the literature review")


st.header("") #create blank space
st.subheader("Related Work v4")
st.write('''
    placeholder text introducing the following tables of work we reviewed
    ''')
c = st.container()
with c:
    c.markdown("").info('''
        *Title:* paper's full title 1 the title of the paper  
        *Topic:* short topic description    
        *Implication for our project:* The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse)  
        ''')
    c.markdown("").info('''
        *Title:* paper's full title 2 the title of the paper  
        *Topic:* short topic description    
        *Implication for our project:* The author uses Linear Regression, Random Forest Regression and a Neural Network to predict the number of upvotes. It ignores the temporal elements of Augury’s study and has a different approach to NLP using Bag of Words, TF-IDF(Term Frequency-Inverse)  
        ''')



st.write("placeholder for text at the end of the literature review")






st.subheader("Ethical Considerations")
st.write('''
    There are clearly ethical implications relating from broadcasting messages on social media and the related investments to which these messages refer, as highlighted by the ‘Reddit Rally’. In project Augury we are only looking at the popularity of posts, and we have not correlated this to market activity, which could be an extension of this work.  To some extent this research has already been investigated in XXX   . Therefore we have made no further mitigations in our project related to market ethics.

Another consideration is the free and open nature of online social media, and Reddit in particular. This can, and does, lead to environments which can become toxic in a number of ways. The subreddits which are looking at Investment are typically more professional in nature so the main way in which toxicity occurs in these fora is through the use of profane language. In our pipeline we have removed this language from our analysis using the python module profanity-filter 1.3.3 which replaces profane words with the character “#”.  

    ''')


st.header("Our Approach")
st.subheader("Project Pipeline")
project_pipeline_image = Image.open('blog_assets/project_pipeline.png')
st.image(project_pipeline_image, caption='Augury Project Pipeline')
st.write('''
    The data scrape we finally put into use called upon the PRAW library to make efficient requests to Reddit’s API.  Our approach to scraping was informed by the following article’s XXXX and the code which we eventually used can be found on the project repo, linked at the head of this page. We knew that we wanted to capture the evolution of a posts popularity over time in order to see how and when popular posts developed, so we designed a pipeline that:
    ''')
st.markdown('''
    - Captured a sample of ‘new posts’ each hour.
        - These were initially five posts per hour, but the consequences of the remainder of our cleaning function meant that this process would ‘timeout’ on AWS Lambda.  So we decided at the end of February 2022 to strip our ‘new posts’ back to one per hour.
    - Tracked these posts over a 24 hour period.
        - Each subsequent scrape would recapture the data related to our target posts, such as the number of comments, the text of those comments, the karma (for which you might assume ‘popularity’) of the comment and post authors.
    - Cleaned the data
        -  Before we stored the data onto our AWS RDS (a postgresql instance) our code did a lot of the early cleaning work for us.  Initially our requests backed libraries wrote large JSON files to our database, when we went into production we had developed code that extracted and formatted the data we wanted from the subreddit’s, making downstream processing of the data much slicker.  For example XXXX 
    - Load the data
        - Our database design/schema of tables is shown below.  This design was to optimize the functioning of the RDS and minimize storage, by reducing duplication to a minimum.


    ''')
db_schema_image = Image.open('blog_assets/db_schema.png')
st.image(db_schema_image, caption='Augury Database Schema (created with website_credit)')
st.markdown('''
    - Extracting and cleaning the data
    - Feature Engineering
        - placeholder
    - modeling
        - placeholder
    - prediction / production
        - placeholder

    ''')




st.header("Testing Reddit Access")
st.markdown("Testing Reddit access...")
### SECRETS TO DELETE ###
# REDDIT_USERNAME= 'Qiopta'
# REDDIT_PASSWORD= 'd3.xr@ANTED#2-L'
# APP_ID= '8oUZoJ3VwfzceEInW3Vd1g'
# APP_SECRET= 'pRg3qU2brsbsyPrPaNP26vxPgwAJbA'
# APP_NAME= 'Capstone2'
### SECRETS TO DELETE ###

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


