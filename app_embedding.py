import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

st.title("📩 Ham/Spam Mail Classification")

df = pd.read_csv("mail_data.csv")

mail_data = df.fillna('')

mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})

@st.cache_resource
def load_resources():
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    X_embeddings = np.load("mail_embeddings.npy")
    return model_st, X_embeddings

embed_model, X_embeddings = load_resources()
Y = mail_data['Category'].astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X_embeddings, Y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')
model.fit(X_train, Y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(Y_train,train_pred)
test_acc = accuracy_score(Y_test,test_pred)

st.write(f" Training Accuracy: {train_acc:.2f}")
st.write(f" Test Accuracy: {test_acc:.2f}")

st.subheader("Tester votre email ici :")

input_mail = st.text_area("Enter email text:")

if st.button("Predict"):
    if input_mail.strip() != "":
        # Transformation 
        input_data_features = embed_model.encode([input_mail])
        # Prediction
        prediction = model.predict(input_data_features)
        # Affichage
        if prediction[0] == 1:
            st.success("✅ Ham Mail (Not Spam)")
        else:
            st.error("🚨 Spam Mail")
    else:
        st.warning("⚠️ Please enter an email")



