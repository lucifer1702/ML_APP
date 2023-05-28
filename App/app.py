import streamlit
import numpy as np
import pandas as pd
import joblib

pipe=joblib.load(open("/Users/mukund/Desktop/IIT_KGP/MLDL/ML_APP/App/models/piped_file_emotions.pkl",'rb'))

def predict_results(docs):
    result=pipe.predict([docs])
    return result[0]
def get_prediction(docs):
    results=pipe.predict_proba([docs])
    return results


def main():
    streamlit.title("emotion classifier app")
    menu = ["home", "Monitor", "about"]
    choice = streamlit.sidebar.selectbox("Menu", menu)
    if choice == "home":
        streamlit.subheader("QUERY  ")
        with streamlit.form(key='emotion-detect'):
            raw_text = streamlit.text_area("type-here")
            sub_text = streamlit.form_submit_button(label='YES')
        if sub_text:
            col1, col2 = streamlit.columns(2)
            prediction=predict_results(raw_text)
            probability=get_prediction(raw_text)


            with col1:
                streamlit.success("original text")
                streamlit.write(raw_text)
                streamlit.success("prediction")
                streamlit.write(prediction)
                streamlit.write("confidence  : {}" .format(np.max(probability)))
            with col2:
                streamlit.success("probability of prediction")
                streamlit.write(probability)

    elif choice == "Monitor":
        streamlit.subheader("monitor-app")
        streamlit.write('this part will be updated later to monitor the status of the website and other cool stuff')
    else:
        streamlit.subheader("About")
        streamlit.write("this is me Mukundan . Writing this, I feel happy That this is the first website I built with STREAMLIT AND MY FIRST ML POWERED WEBSITE")


if __name__ == '__main__':
    main()
