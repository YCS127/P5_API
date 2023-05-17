import streamlit as st
import pandas as pd
import numpy as np

from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from utils import clean_html, text_cleaning, tokenize, filtering_nouns, lemmatize, LdaModel, SupervisedModel

def main():
    st.title("Application de machine learning pour catégoriser automatiquement des questions")
    st.markdown("**OpenClassrooms** Projet n°5 du parcours Machine Learning réalisé en mai 2023")
    st.info("Auteur: Claude Sabardeil")
    st.markdown("_"*10)
    
    if st.checkbox("Afficher le détail de la procédure", key=False):
            st.markdown("#### Prédiction de tags effectuée en utilisant un modèle \
                        de classification supervisée et un modèle non supervisée")
            st.markdown("* **Modèle supervisé: OneVsRestClassifier(LinearSVC())**")
            st.markdown("* **Modèle non supervisé: LDA model**")
    
    #cust_input = str(st.text_input("**Saisissez votre question**"))
    cust_input = st.text_area("Write your text")
    
    
    
    if st.button("Exécuter la prédiction de tags"):
        if len(cust_input) !=0:
        
            #on prepare le texte
            text_wo_html = clean_html(cust_input)
            cleaned_text = text_cleaning(text_wo_html)
            tokenized_text = tokenize(cleaned_text)
            filtered_noun_text = filtering_nouns(tokenized_text)
            lemmatized_text = lemmatize(filtered_noun_text)
            lda_model = LdaModel()
            unsupervised_pred = list(lda_model.predict_tags(lemmatized_text))
            supervised_model = SupervisedModel()
            supervised_pred = list(supervised_model.predict_tags(lemmatized_text))
        

            tag_full = set(unsupervised_pred + supervised_pred)
            
            # afficher les résultats
            if len(tag_full) != 0:
                st.markdown("#### - Predicted tags")
                for elt in tag_full:
                    if (elt in supervised_pred) & (elt in unsupervised_pred):
                        st.markdown("<mark style='background-color: lightgray'>**" + elt + "**</mark>",
                                    unsafe_allow_html=True)
                
                    if (elt in supervised_pred) & (elt not in unsupervised_pred):
                        st.markdown("<mark style='background-color: lightblue'>**" + elt + "**</mark>",
                                    unsafe_allow_html=True)
                    
                    if (elt not in supervised_pred) & (elt in unsupervised_pred):
                        st.markdown("<mark style='background-color: lightgreen'>**" + elt + "**</mark>",
                                    unsafe_allow_html=True)
                    
                st.markdown("")
                st.markdown("<mark style='background-color: lightgray'>""</mark> &nbsp;Both models have predicted",
                            unsafe_allow_html=True)
                st.markdown("<mark style='background-color: lightblue'>""</mark> &nbsp;Only supervised model has predicted",
                            unsafe_allow_html=True)
                st.markdown("<mark style='background-color: lightgreen'>""</mark> &nbsp;Only unsupervised model has predicted",
                            unsafe_allow_html=True)
            else:
                st.markdown("#### Aucun tag prédit")
              
        else:
            st.info("Please, write your text before trying 'extraction tags'!")
        
         
            
        
    
    
if __name__ == '__main__':
        main()