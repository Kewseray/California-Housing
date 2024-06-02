import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# import dataset

data = pd.read_csv('California_Houses.csv')

# Split datas
X = data[['Median_Income', 'Tot_Rooms', 'Distance_to_coast']]
y = data['Median_House_Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
linear = LinearRegression()
linear.fit(X_train, y_train)

RFregressor = RandomForestRegressor()
RFregressor.fit(X_train, y_train)

print("Test R2 - LR: {}".format(linear.score(X_test,y_test)))
print("Test R2 - RF: {}".format(RFregressor.score(X_test,y_test)))

# Application
st.set_page_config(
    page_title="Predictions",
    page_icon="🖥",
    layout="wide"
)

hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# left bar
with st.sidebar:
    st.markdown("<h1 style='color: white;text-align: center; margin-bottom: 50px'>Choix du modèle</h1>",
                unsafe_allow_html=True)

    m_dropdown = st.selectbox('Modèle', ['Régression linéaire',  'Random Forest'])

    slider = st.slider('Pourcentage de données utilisé pour prédire', 0, 100, 25)

# menu
st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 50px'>Predictions</h1>",
            unsafe_allow_html=True)

# Use st.column instead of st.columns
col1 = st.columns(1)

with col1[0]:
    st.subheader('Prix de maison')

    form = st.form(key="house_value_prediction_form")

    feature1 = form.number_input('Median_Income', value=0)
    feature2 = form.number_input('Tot_Rooms', value=0)
    feature3 = form.number_input('Distance_to_coast', value=0)
    button = form.form_submit_button("Predict")

    df = pd.DataFrame({'Median_Income': [feature1], 'Tot_Rooms': [feature2], 'Distance_to_coast': [feature3]})

    if button:
        if m_dropdown == 'Régression linéaire':
            pred = linear.predict(df)
        elif m_dropdown == 'Random Forest':
            pred = RFregressor.predict(df)
        st.write('Les prix de maison sont estimés à :', pred)

st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 50px; margin-top: 50px'>Graphiques</h1>",
            unsafe_allow_html=True)
