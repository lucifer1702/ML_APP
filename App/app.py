import streamlit
import numpy as np 
import pandas as pd 
import joblib

def main():
    menu=["home","Monitor","about"]
    choice=streamlit.sidebar.selectbox("Menu",menu)
    

if __name__=='__main__':
    main()    