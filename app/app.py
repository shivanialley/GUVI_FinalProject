import streamlit as st
st.set_page_config(
    page_title="Bank Deposit Predictor",
    page_icon="🏦",
    layout="centered"
)

st.markdown("### 💼 Bank Term Deposit Prediction System")
st.markdown("MLflow-tracked | XGBoost | AWS Deployed")

with st.form("prediction_form"):
    age = st.slider("Age", 18, 95, 35)
    balance = st.number_input("Account Balance", -5000, 200000, 1000)
    campaign = st.slider("Campaign Contacts", 1, 50, 1)
    previous = st.slider("Previous Contacts", 0, 50, 0)

    submit = st.form_submit_button("Predict")

if submit:
    X = pd.DataFrame([{
        "age": age,
        "balance": balance,
        "campaign": campaign,
        "previous": previous
    }])

    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    st.progress(int(prob * 100))

    st.metric("Subscription Probability", f"{prob:.2%}")
