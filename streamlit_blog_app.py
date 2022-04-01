import streamlit as st
#import pandas as pd
#import joblib
#from sklearn.linear_model import LogisticRegression


# Title
st.header("Team Augury Capstone Project")


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