import streamlit
import numpy as np
import pandas as pd
import joblib

import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from sklearn.ensemble import VotingClassifier
import plotly.figure_factory as ff

model1=load_model('App/models/model_lstm.h5')
model2=joblib.load('App/models/piped_file_emotions.pkl')



cols=['anger', 'boredom', 'empty', 'enthusiasm', 'fun', 'happiness', 'hate',
       'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']

def tokenf(sample):
    sample=pd.Series(sample)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sample)
    word_indices = tokenizer.texts_to_sequences(sample)
    word_index = tokenizer.word_index
    x_sample=pad_sequences(word_indices,maxlen=100)
    return x_sample




def predict_results(docs):
    docs= tokenf(docs)
    ##voting ensemble 
    result=model1.predict(docs)
    return result 



def main():
    streamlit.title("emotion classifier app")
    menu = ["home", "Monitor", "about"]
    choice = streamlit.sidebar.selectbox("Menu", menu)
    if choice == "home":
        streamlit.subheader("QUERY")
        with streamlit.form(key='emotion-detect'):
            raw_text = streamlit.text_area("type-here")
            sub_text = streamlit.form_submit_button(label='YES')
        if sub_text:
            col1, col2 = streamlit.columns(2)
            prediction=predict_results(raw_text)
            emotion=cols[np.argmax(prediction)]
            probability=np.max(predict_results(raw_text))
           


            with col1:
                streamlit.success("original text")
                streamlit.write(raw_text)
                streamlit.success("prediction")
                streamlit.write(emotion)
                streamlit.write("confidence  : {}" .format(np.max(probability)))
            with col2:
                streamlit.success("probability of each class prediction")
                # streamlit.write(probability)
                proba_df =pd.DataFrame(prediction ,columns=cols)
                streamlit.write(proba_df)
                # fig = ff.create_distplot(prediction,cols, bin_size=[.1, .25, .5])
                # streamlit.plotly_chart(fig, use_container_width=True)

                
    elif choice == "Monitor":
        streamlit.subheader("monitor-app")
        streamlit.write('this part will be updated later to monitor the status of the website and other cool stuff')

    else:
        streamlit.subheader("About")
        streamlit.write("this is me Mukundan . Writing this, I feel happy That this is the first website I built with STREAMLIT AND MY FIRST ML POWERED WEBSITE")


if __name__ == '__main__':
    main()
