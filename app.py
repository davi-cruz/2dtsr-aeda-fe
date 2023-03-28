import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import streamlit as st


st.markdown('# Aplicação e análise exploratória de dados')
file = st.file_uploader('Faça o upload do arquivo de consumidor', type='csv')

if file:
    st.markdown("## Dataframe de consumidor")
    df_consumidor = pd.read_csv(file, sep=';', encoding='latin1')
    st.dataframe(df_consumidor.head(10))

    file2 = st.file_uploader('Faça o upload do arquivo do governo', type='csv')
    if file2:
        st.markdown("## Dataframe do governo")
        df_governo = pd.read_csv(file2, sep=';', encoding='latin1')
        st.dataframe(df_governo.head(10))
    st.sidebar.markdown("## Escolha a página")