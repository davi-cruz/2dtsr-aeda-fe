import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

def main():
    st.markdown('# Esta é uma aplicação de análise exploratória')

    file = st.file_uploader("Faça o upload do arquivo do consumidor", type=['CSV'])

    if file:
        st.markdown("## Dataframe do consumidor")
        df_consumidor = pd.read_csv(file, sep=';', encoding='ISO-8859-1')
        st.dataframe(df_consumidor.head())

        file2 = st.file_uploader("Faça o upload do arquivo do governo", type=['CSV'])

        if file2:

            st.markdown("## Dataframe do governo")
            df_governo = pd.read_csv(file2, sep=';', encoding='ISO-8859-1')
            st.dataframe(df_governo.head())

            df_governo = df_governo.rename(columns={'PRINCÍPIO ATIVO': 'SUBSTÂNCIA'})

            st.markdown("## Alterando o nome da coluna")
            st.dataframe(df_governo.head())

            st.markdown("## Verificando os valores nulos do df_governo")
            st.table(df_governo.isnull().sum())

    #df_governo = pd.read_csv('TA_PRECO_MEDICAMENTO_GOV.CSV', sep=';', encoding='ISO-8859-1')


if __name__ == '__main__':
    main()