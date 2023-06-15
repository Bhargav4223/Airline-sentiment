# This review.py is the user interface provided by the streamlit.

import streamlit as st
import numpy as np
from prediction import predict
st.title("Air Line Sentiment Analysis")#Title of the page
st.markdown("This gives you an actual Sentiment of the customers like whether the given is positive or negative")#Description

review = st.text_input("Kindly Enter your Review of this Air Lines")#User input

st.text('')
if st.button("Review"):#User Review Button
	result = predict(
		np.array([review]))
	st.text(result[0])
		

st.text('')#This shows up the Review as positive,negative or neutral.
st.text('')
