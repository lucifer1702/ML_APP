# ML_APP

This is a sentiment analyser app made with streamlit

The app will contain 2 directories :

1. App folder which contains a models folder which holds the pickled model
2. The notebook folder will contain  the notebook which will be used for the training of the ML model and also a dataset folder which has datasets in it.

Since the accuracy of classical Ml models were really poor in the case of both tfidf and count vectorizer I decided to use DL Models . But due some dependancy issues I am currently unable to use them so we would be sticking with Ml models for now . Will update it later in the future with DL mode

To run the app

   1.Clone the repo using git clone

2. cd into the directory ML_app
3. then run the command

   ```
   streamlit run App/app.py
   ```
   4. The website must run on your local host
