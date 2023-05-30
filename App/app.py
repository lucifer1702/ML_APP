import streamlit
import numpy as np
import pandas as pd
import joblib
import altair as alt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

model=load_model('App/models/model_lstm.h5')

class_dict={0:'anger', 1:'boredom', 2:'empty', 3:'enthusiasm', 4:'fun', 5:'happiness', 6:'hate',
       7:'love', 8:'neutral', 9:'relief',10: 'sadness', 11:'surprise', 12:'worry'
}

max_fatures = 5000

def tokenf(X):
    token=Tokenizer(split=' ',num_words=max_fatures)
    token.fit_on_texts([X])
    X = token.texts_to_sequences([X])
    X = pad_sequences(X,maxlen=80)

    return X


def predict_results(docs):
    docs= tokenf(docs)
    result=model.predict(docs)
   
    return result



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
            emotion=class_dict[np.argmax(prediction)]
            probability=np.max(predict_results(raw_text))


            with col1:
                streamlit.success("original text")
                streamlit.write(raw_text)
                streamlit.success("prediction")
                streamlit.write(emotion)
                streamlit.write("confidence  : {}" .format(np.max(probability)))
            with col2:
                streamlit.success("probability of each class prediction")
                streamlit.write(probability)
                proba_df =pd.DataFrame(prediction ,columns=class_dict.values)
                streamlit.write(proba_df)

                fig =alt.Chart(proba_df).mark_bar().encode(x='Emotions',y='probability',color='Emotions')
                streamlit.altair_chart(fig)

    elif choice == "Monitor":
        streamlit.subheader("monitor-app")
        streamlit.write('this part will be updated later to monitor the status of the website and other cool stuff')

    else:
        streamlit.subheader("About")
        streamlit.write("this is me Mukundan . Writing this, I feel happy That this is the first website I built with STREAMLIT AND MY FIRST ML POWERED WEBSITE")


if __name__ == '__main__':
    main()
