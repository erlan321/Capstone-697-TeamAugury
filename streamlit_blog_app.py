import streamlit as st
#import pandas as pd
#import joblib
#from sklearn.linear_model import LogisticRegression


# Title
st.header("Team Augury Capstone Project")
st.markdown("Welcome a sample of the blog space for our project.")

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


