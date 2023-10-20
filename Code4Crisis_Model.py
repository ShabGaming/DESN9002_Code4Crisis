import streamlit as st
import pandas as pd
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pickle
import warnings
warnings.filterwarnings('ignore')

def standardize_text(text):
    text = text.lower()
    text = text.replace(r'[^\w\s]+', '')
    text = text.replace(r'-', ' ')
    text = text.replace(r'\'s', '')
    return text

KeyWordsList = ["nagorno karabakh conflict","nagorno karabakh","armenia azerbaijan","azerbaijan","armenia","fighting","propaganda","armenia military","azerbaijan military","armenia and azerbaijan","shelling","conflict","civilians","war","deaths","armenian genocide","territorial integrity","nagorno_karabakh","hostilities","caucasus","ilham aliyev","south caucasus","nikol pashinyan","georgia","ceasefire","russia","armenia azerbaijan clashes","clashes","collective security treaty organization","de escalation","civilian","skirmishes","military positions","defense","diplomatic solution","soldiers","refugees","peace talks","stability","nagorno karabakh tensions","western powers","military training","garegin nzhdeh","nationalism","affect","arms exports","rocket launchers","geopolitica","conspiracy theories","military","media","soviet union","drone","drones","united states","self determination","border","tavush region","artillery","compromise","nagorno karabakh war","rhetoric","geopolitics","minsk group","turkey","united nations","csto","armenia azerbaijan conflict","oil reserves","serbia","opposition parties","disinformation","frontline","authorities","russian empire","children","separatist region","aircraft","tanks","ethnic armenian forces","lachin corridor","disarmament","separatists","diplomacy","sovereignty","aliyev","resilience","paris peace conference","diplomatic steps","artsakh","ethnic armenians","european union","geopolitical interests","mediation","armenian forces","war crime","occupation","humanitarian ceasefire","osce","violation","truce","smerch missiles","troika","cluster bombs","mortars","nagorno karabakh autonomous oblast","cease fire","envoys","buffer zone","atrocities"]

def converting_text_to_armenia_azerbaijan(text, country1, country2, region):
    text = text.lower()
    for i in text:
        if i == country1:
            text = text.replace(i, 'armenia')
        elif i == country2:
            text = text.replace(i, 'azerbaijan')
        elif region != None and i == region:
            text = text.replace(i, 'nagorno Kkarabakh')
        elif i == country1 + 's':
            text = text.replace(i, 'armenias')
        elif i == country2 + 's':
            text = text.replace(i, 'azerbaijans')
        elif i == region + 's' and region != None:
            text = text.replace(i, 'nagorno karabakhs')
        elif i == country1 + 'n':
            text = text.replace(i, 'armenian')
        elif i == country2 + 'n':
            text = text.replace(i, 'azerbaijani')
        elif i == country1 + 'i':
            text = text.replace(i, 'armenian')
        elif i == country2 + 'i':
            text = text.replace(i, 'azerbaijani')
    return text

sid = SIA()
def perform_sentiment_analysis(text):
    sentiment_scores = sid.polarity_scores(text)
    sentiment_scores = sentiment_scores['compound']
    if sentiment_scores >= 0.06:
        return 1
    elif sentiment_scores <= 0.035:
        return -1
    else:
        return 0

def run_model(text, country1, country2, region, keywords):
    print("Standardizing text...")
    text = standardize_text(text)
    country1 = standardize_text(country1)
    country2 = standardize_text(country2)
    region = standardize_text(region)
    print("Converting text to Armenia and Azerbaijan...")
    text = converting_text_to_armenia_azerbaijan(text, country1, country2, region)
    print("Extracting features...")
    columns = ['sentiment', 'word count']
    columns.extend(keywords)
    row = []
    row.append(perform_sentiment_analysis(text))
    row.append(len(text.split()))
    for i in range(len(keywords)):
        row.append(0)
    for index in range(2, len(columns)):
        if columns[index] in text:
            row[index] = row[index] + 1
    print("Loading model...")
    model = pickle.load(open('temp_files/logistic_regression_model.pkl', 'rb'))
    scaler = pickle.load(open('temp_files/scaler.pkl', 'rb'))
    print(row)
    print("Scaling data...")
    row = scaler.transform([row])
    print("Predicting...")
    prediction = model.predict(row)
    print("Done!")
    return prediction[0]


st.set_page_config(
    page_title="C4C Model",
    page_icon="🤖",
)

st.title("Code4Crisis Model ⚔️")
st.sidebar.markdown("# Use Model 🤖")

text = st.text_input("Enter your text here:")
st.caption("Reddit, Twitter, Facebook, News etc.")
country1 = st.text_input("Enter country 1:")
st.caption("For example, Azerbaijan")
country2 = st.text_input("Enter country 2:")
st.caption("For example, Armenia")
region = st.text_input("(optional) Enter region/conflict name:")
st.caption("For example, Nagorno Karabakh")

if st.button("Run Model", type="primary"):
    if text == "":
        st.write("❌ Please enter text ❌")
        exception = True
    elif country1 == "":
        st.write("❌ Please enter country 1 ❌")
        exception = True
    elif country2 == "":
        st.write("❌ Please enter country 2. ❌")  
        exception = True  
    if region == "":
        region = None
    if exception != True:
        x = run_model(text, country1, country2, region, KeyWordsList)
        if x == 1:
            st.write("Crisis may have started or is about to start.")
        elif x == -1:
            st.write("A crisis have not been detected.")
        else:
            st.write("Error in running model.")

st.divider()
st.caption("Made by code4crisis")
