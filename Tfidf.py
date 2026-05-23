import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#  Titre de l'application
st.title("📩 Ham/Spam Mail Classification")

df = pd.read_csv(r"mail_data.csv")

mail_data = df.fillna('')

mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})


X = mail_data['Message']
Y = mail_data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')

model.fit(X_train_features, Y_train)

train_pred = model.predict(X_train_features)
test_pred = model.predict(X_test_features)

train_acc = accuracy_score(Y_train, train_pred)
test_acc = accuracy_score(Y_test, test_pred)

st.write(f" Training Accuracy: {train_acc:.2f}")
st.write(f" Test Accuracy: {test_acc:.2f}")

#  Interface utilisateur

st.subheader("Tester votre  email ici :")

input_mail = st.text_area("Enter email text:")

if st.button("Predict"):
    if input_mail.strip() != "":
        
        # transformation
        input_data_features = feature_extraction.transform([input_mail])
        
        # prediction
        prediction = model.predict(input_data_features)

        # affichage
        if prediction[0] == 1:
            st.success("✅ Ham Mail (Not Spam)")
        else:
            st.error("🚨 Spam Mail")
    else:
        st.warning("⚠️ Please enter an email") 