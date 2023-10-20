import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="C4C Documentation",
    page_icon="📚",
)

st.sidebar.markdown("# Documentation 📚")

st.title("Code4Crisis ⚔️")
st.header("Detection of the emergence of armed conflicts through short-term land disputes.", divider='red')
st.markdown('The project is trained on Reddit & New York Times posts.')

st.markdown("### Data Collection 📊")
st.markdown("Keywords found using GPT 4 (LLM) from across news articles.")
KeyWordsCSV = pd.read_csv('KeyWords.csv')
KeyWordsDF = pd.DataFrame(KeyWordsCSV)
KeyWordsDF = KeyWordsDF.sort_values(by=['Search Volume'], ascending=False)
KeyWordsDF = KeyWordsDF.drop(KeyWordsDF.columns[0], axis=1)
KeyWordsDF = KeyWordsDF.dropna()
KeyWordsDF = KeyWordsDF[KeyWordsDF['Search Volume'] >= 0.15]


def prepend_country(row):
    if ("Armenia" not in row[0]) and ("Azerbaijan" not in row[0]):
        return "Armenia " + row[0]
    else:
        return row[0]


def prepend_country2(row):
    if ("Armenia" not in row.iloc[0]) and ("Azerbaijan" not in row.iloc[0]):
        return "Azerbaijan " + row.iloc[0]
    else:
        return row.iloc[0]


KeyWordsDF_temp = KeyWordsDF.copy()
KeyWordsDF_temp['Keyword'] = KeyWordsDF_temp.apply(prepend_country, axis=1)
KeyWordsDF_temp2 = KeyWordsDF.copy()
KeyWordsDF_temp2['Keyword'] = KeyWordsDF_temp2.apply(prepend_country2, axis=1)
KeyWordsDF2 = pd.concat([KeyWordsDF_temp, KeyWordsDF_temp2])
KeyWordsDF2 = KeyWordsDF2.drop_duplicates(subset=['Keyword'], keep='first')
KeyWordsDF2 = KeyWordsDF2.reset_index(drop=True)
st.dataframe(KeyWordsDF2, hide_index=True)
st.caption("Word Cloud of Keywords")
st.image("resources/output.png")
st.caption("Word Cloud of Chosen Keywords")
st.image("resources/output2.png")
# Drop down menu for choosing API
API = st.selectbox("Choose API", ["Reddit", "New York Times", "Combined"])
nyt = pd.read_csv(r"resources/NYT.csv")
reddit = pd.read_csv(r"resources/Reddit.csv")
combined = pd.read_csv(r"temp_files/df.csv")
if API == "Reddit":
    st.caption("Data scraped from Reddit, (Double click to expand a block)")
    st.dataframe(reddit, hide_index=True)
elif API == "New York Times":
    st.caption(
        "Data scraped from New York Times, (Double click to expand a block)")
    st.dataframe(nyt, hide_index=True)
elif API == "Combined":
    st.caption(
        "Data scraped from both Reddit and New York Times, (Double click to expand a block)")
    st.dataframe(combined, hide_index=True)

st.divider()
st.markdown("### Data Cleaning & Processing 🧹")
st.markdown("Using an open-source library called Llama-2 7b, we are implementing quantization to reduce the bit depth to 4 bits in order to achieve faster response times. While this sacrifice may affect the quality of the response, it is acceptable since the response required for our use case is a binary value. Additionally, we have developed a prompt to classify each title and determine its relevance to our topic and target audience.")
st.markdown("Initially, we conducted tests using other Llama models, specifically the 13b model fine-tuned for instructions. However, the average response time for these models was approximately 30 seconds. In contrast, our current model yields a response time of around 3 seconds, which is significantly faster. This is a must as we have to run it on each row (roughly 4,500 rows)")

st.markdown("Code for Llama-2 7b, quantization")

code = '''model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.,
    top_p=0.95,
    repetition_penalty=1.15)'''
st.code(code, language='python')
st.markdown("Prompt Engineered: Your job is to determine if the topic is related to the Armenia Conflict/war with Azerbaijan over the Nagorno-Karabakh region. If the topic is related to the Armenia Conflict/war with Azerbaijan, please type 'yes'. If the topic is not related to the Armenia Conflict/war with Azerbaijan, please type 'no'. Only answer 'yes' or 'no' to the question. ANSWER ONCE with ONE WORD")
df_temp = pd.read_csv(r"temp_files/df_cleaned.csv")
st.caption("Dataframe after cleaning, and removing outliers (< 2020)")
st.dataframe(df_temp, hide_index=True, column_order=(
    "Date", "related to conflict", "Topic", "Sub Topic"))

st.divider()
st.markdown("### Feature Engineering 🛠️")
st.markdown("We used 2 techniques to generate features for the dataset")
Technique = st.selectbox("Choose Technique", [
                         "Sentiment Analysis", "Keyword Count"])
if Technique == "Sentiment Analysis":
    st.markdown("Using the Vader Sentiment Analysis library, we were able to generate a sentiment score for each title. The sentiment score is a number between -1 and 1, with -1 being the most negative and 1 being the most positive.")
    code2 = '''import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    sid = SIA()

    df_cleaned['sentiment'] = np.nan

    for index, row in df_cleaned.iterrows():
        sentiment = sid.polarity_scores(row['Topic'])
        if sentiment['compound'] >= 0.06:
            df_cleaned.loc[index, 'sentiment'] = 1
        elif sentiment['compound'] <= 0.035:
            df_cleaned.loc[index, 'sentiment'] = -1
        else:
            df_cleaned.loc[index, 'sentiment'] = 0'''
    st.code(code2, language='python')
elif Technique == "Keyword Count":
    st.markdown("Using the keywords found earlier, we were able to generate a binary value for each title, each keyword was given a column and if the title contained the keyword, the value would be +1")
    code3 = '''df_final['word count'] = df_final['Topic'].str.split().str.len()

for word in KeyWordsList:
    df_final[word] = 0

for index, row in df_final.iterrows():
    for word in KeyWordsList:
        df_final.loc[index, word] = row['Topic'].count(word)

df_final = df_final.loc[:, (df_final != 0).any(axis=0)]
'''
    st.code(code3, language='python')
st.markdown("Final Dataframe")
df_final = pd.read_csv(r"temp_files/df_final.csv")
st.caption("Crisis represents our target variable")
st.dataframe(df_final, hide_index=True)

st.divider()
st.markdown("### Model Training & Evaluation 🧠")
st.markdown(
    "We used 3 different models to train our dataset, and compared the results to find the best model.")
Model = st.selectbox(
    "Choose Model", ["Logistic Regression", "Random Forest", "Neural Network"])
if Model == "Logistic Regression":
    code4 = '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
df = pd.read_csv('temp_files/df_final.csv')

# Define the features (X) and the target (y)
X = df.drop(columns=['crisis'])
y = df['crisis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
logistic_reg = LogisticRegression(random_state=42)

# Fit the model to the training data
logistic_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_reg.predict(X_test)

# Evaluate the logistic regression model
accuracy_lr = accuracy_score(y_test, y_pred)
classification_report_result_lr = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy_lr:.2f}")
print("Classification Report:")
print(classification_report_result_lr)'''
    st.code(code4, language='python')
    st.text('''
                Accuracy: 0.73
                Classification Report:
                                precision   recall   f1-score  support

                        0.0       0.78      0.08      0.14        91
                        1.0       0.72      0.99      0.84       223

                accuracy                              0.73       314
                macro avg         0.75      0.53      0.49       314
                weighted avg      0.74      0.73      0.64       314
                ''')
elif Model == "Random Forest":
    code5 = '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
df = pd.read_csv('temp_files\df_final.csv')

# Define the features (X) and the target (y)
X = df.drop(columns=['crisis'])
y = df['crisis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred)
classification_report_result_rf = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy_rf:.2f}")
print("Classification Report:")
print(classification_report_result_rf)'''
    st.code(code5, language='python')
    st.text('''
                Accuracy: 0.67
                Classification Report:
                                precision   recall   f1-score  support

                        0.0       0.40      0.30      0.34        91
                        1.0       0.74      0.82      0.78       223

                accuracy                              0.67       314
                macro avg         0.57      0.56      0.56       314
                weighted avg      0.64      0.67      0.65       314
                ''')
elif Model == "Neural Network":
    code6 = '''from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report
from keras.callbacks import EarlyStopping

# Load your dataset
df = pd.read_csv('temp_files/df_final.csv')

# Define the features (X) and the target (y)
X = df.drop(columns=['crisis'])
y = df['crisis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

# Add input layer and hidden layers
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))  # Adjust the number of units as needed
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))

# Add output layer (sigmoid activation for binary classification)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=60)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=350, batch_size=4, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Make predictions on the test data
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Evaluate the neural network model
accuracy_nn = accuracy_score(y_test, y_pred)
classification_report_result_nn = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy_nn:.2f}")
print("Classification Report:")
print(classification_report_result_nn)'''
    st.code(code6, language='python')
    st.caption(
        "Early Stopping Callback enabled, with 350 epochs and batch size of 4")
    st.text('''
                accuracy                              0.71       314
                macro avg         0.63      0.57      0.57       314
                weighted avg      0.68      0.71      0.68       314
                ''')

st.divider()
# Graph of accuracy of each model
st.caption("Accuracy of each model")
df_bar = pd.DataFrame({'Accuracy': [0.73, 0.67, 0.71], 'Model': [
                      "Logistic Regression", "Random Forest", "Neural Network"]})
st.bar_chart(df_bar, x="Model", y="Accuracy", color="Accuracy")
