import pickle
import streamlit as st

with open('model.pkl', 'rb') as file:
    knn_model = pickle.load(file)
    


st.title('Iris Machine Learning Model,')

sepal_l = st.number_input('Sepal length')
sepal_w = st.number_input('Sepal Width')
petal_l = st.number_input('Petal length')
petal_w = st.number_input('Petal Width')

user_data = [sepal_l, sepal_w, petal_l, petal_w]


if st.button('Make Prediction on your data!!!'):
    prediction = knn_model.predict([user_data])
    st.subheader('Your flower is:', prediction[0])
    st.subheader(prediction[0])
    st.balloons()
    