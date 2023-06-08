import streamlit as st
import numpy as np

from prediction import predict
st.title("Air Line Sentiment Analysis")
st.markdown("This gives you an actual Sentiment of the customers like whether the given is positive or negative")

review = st.text_input("Kindly Enter your Review of this Air Lines")

st.text('')
if st.button("Review"):
	result = predict(
		np.array([review]))
	st.text(result[0])
		

st.text('')
st.text('')
