import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import streamlit as st
import warnings
import re

warnings.filterwarnings('ignore')

# Define as configurações de exibição
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def main():
    st.markdown('# 2DTSR - Advanced EDA - Atividade')
        
    # Importa data sources
    df_consumidor = pd.read_csv('TA_PRECO_MEDICAMENTO.CSV', sep=';', encoding='ISO-8859-1', decimal=",")
    df_governo = pd.read_csv('TA_PRECO_MEDICAMENTO_GOV.CSV', sep=';', encoding='ISO-8859-1', decimal=",")

    st.markdown("**Insights**: Uma vez que durante a análise pré-importação dos dados era sabido que os valores numericos utilizavam a vírgula como separador decimal, ajustado item durante importação no dataframe para simplificar, utilizando o parametro 'decimal=","', assim como o encoding 'ISO-8859-1` foi utilizado devido à acentuação e caracteres latinos.")
    code = '''# Importa data sources
    df_consumidor = pd.read_csv('TA_PRECO_MEDICAMENTO.CSV', sep=';', encoding='ISO-8859-1', decimal=',')
    df_governo = pd.read_csv('TA_PRECO_MEDICAMENTO_GOV.CSV', sep=';', encoding='ISO-8859-1', decimal=',')
    '''
    st.code(code, language='python')    

    st.markdown('### Lista primeiras linhas do df_consumidor')
    st.dataframe(df_consumidor.head(3))

    st.markdown('### Lista primeiras linhas do df_governo')
    st.dataframe(df_governo.head(3))

    st.markdown("### Tamanho dos datasets:")
    st.markdown(f"- Dataset de consumidor: ~**{df_consumidor.shape[0]}** medicamentos, contendo **{df_consumidor.shape[1]}** colunas")
    st.markdown(f"- Dataset do governo: ~**{df_governo.shape[0]}** medicamentos, contendo **{df_governo.shape[1]}** colunas")

    st.markdown('**Insights**: Observa-se que o dataset do preços para o governo apresenta uma coluna a mais que o dataset do preço para o consumidor')

    st.markdown('# Verificando os tipos de dados do dataframe consumidor')
    st.write(df_consumidor.info())

    st.markdown('# Verificando os tipos de dados do dataframe governo')
    st.write(df_governo.info())

    st.markdown('**Insights**: Nota-se que todos os dados foram importados com o tipo object, com exceção dos dados numéricos, corrigidos a partir do uso do parametro de separador decimal, o que sobrescreve o padrão utilizado pelo pandas.')

    

    # # %% [markdown]
    # # ### Verificando duplicatas

    # # %%
    # printmd(f"- Dataset do consumidor\n  - Quantidade de linhas: **{len(df_consumidor)}**\n  - Quantidade de registros únicos: **{len(df_consumidor.drop_duplicates().index)}**")
    # printmd(f"- Dataset do governo\n  - Quantidade de linhas: **{len(df_governo)}**\n  - Quantidade de registros únicos: **{len(df_governo.drop_duplicates().index)}**")

    # # %% [markdown]
    # # **Insights**: Observa-se que ambos os datasets possuem registros duplicados, que pode estar relacionado à linhas em branco ou de fato registros duplicado, os quais serão removidos conforme comandos executados na sequencia abaixo.

    # # %%
    # # Remove linhas duplicadas
    # df_consumidor.drop_duplicates(inplace=True)
    # df_governo.drop_duplicates(inplace=True)

    # # %% [markdown]
    # # ### Comparando as colunas entre os dataframes a serem cruzados

    # # %%
    # # Comparando as colunas dos dataframes
    # diff1 = set(df_consumidor.columns) - set(df_governo.columns)
    # diff2 = set(df_governo.columns) - set(df_consumidor.columns)

    # printmd("Itens presentes no `df_consumidor` e não presentes no `df_governo`:")
    # for item in diff1:
    #     printmd(f"- {item}")
    # printmd("")
    # printmd("Itens presentes no `df_governo` e não presentes no `df_consumidor`:")
    # for item in diff2:
    #     printmd(f"- {item}")

    # # %% [markdown]
    # # **Insights**:
    # # - Conforme pode ser observado nos dicionários dos dados, com exceção das colunas **PMVG** (Preço Médio de Venda ao Governo) e **PMC** (Preço Médio ao Consumidor), observa-se que a coluna **PRINCÍPIO ATIVO** possui as mesmas informações que a coluna **SUBSTÂNCIA** do dataframe de consumidor, onde se faz necessário padronizar o nome para simplificar o processo de merge futuro.

    # # %%
    # # Renomeia PRINCÍPIO ATIVO para SUBSTÂNCIA
    # df_governo.rename(columns={'PRINCÍPIO ATIVO': 'SUBSTÂNCIA'}, inplace=True)

    # # %% [markdown]
    # # ### Tratamento de valores nulos

    # # %%
    # # Dataframe consumidor
    # df_consumidor.isnull().sum()
    # msno.matrix(df_consumidor)

    # # %%
    # # Dataframe consumidor
    # df_governo.isnull().sum()
    # msno.matrix(df_governo)

    # # %% [markdown]
    # # **Insights**:
    # # - Observa-se que a grande maioria dos valores nulos estão relacionados a valores de PF (Preço Fábrica), PMVG (Preço Máximo de Venda ao Governo) e PMC (Preço Máximo ao Consumidor).
    # # - Assumindo que o ICMS é o imposto relacionado aos percentuais nas variáveis mencionadas, vide explicação presente nos dicionários de dados, os valores serão diretamente proporcionais e, consequentemente, redundantes na base, onde seria possível obter o valor com imposto de acordo com a região onde o medicamento será comercializado, inclusive em áreas de livre comércio (ALC).
    # # - Portanto, além destes valores, alguns dados de identificação simples dos medicamentos serão removidos, conforme lista abaixo:
    # #   - `CÓDIGO GGREM`: Identificação da apresentação do medicamento
    # #   - `APRESENTAÇÃO`: Apresentação do documento
    # #   - `REGISTRO`: Registro do medicamento junto à Anvisa
    # #   - `EAN 1`, `EAN 2` e `EAN 3`: Códigos de Barras
    # #   - `ANÁLISE RECURSAL`: Se o produto ainda tem análise de preço em curso
    # #   - `COMERCIALIZAÇÃO 2022`: Se medicamento foi comercializado em 2022

    # # %%
    # # Define lista das colunas a serem removidas
    # columns_to_remove = ['CÓDIGO GGREM', 'EAN 1', 'EAN 2', 'EAN 3','ANÁLISE RECURSAL', 'COMERCIALIZAÇÃO 2022', 'PF 12%', 'PF 17%', 'PF 17% ALC', 'PF 17,5%',
    #                     'PF 17,5% ALC', 'PF 18%', 'PF 18% ALC', 'PF 20%', 'PMC 12%', 'PMC 17%', 'PMC 17% ALC', 'PMC 17,5%', 'PMC 17,5% ALC', 'PMC 18%',
    #                     'PMC 18% ALC', 'PMC 20%', 'PMVG 12%', 'PMVG 17%', 'PMVG 17% ALC', 'PMVG 17,5%', 'PMVG 17,5% ALC', 'PMVG 18%', 'PMVG 18% ALC', 'PMVG 20%']

    # # %%
    # # Filtra dataset com base na lista de colunas a serem removidas
    # df_consumidor_filtrado = df_consumidor[set(df_consumidor.columns) - set(columns_to_remove)]
    # df_governo_filtrado = df_governo[set(df_governo.columns) - set(columns_to_remove)]

    # # %%
    # # Analisa Missing Values para df_consumidor_filtrado
    # msno.matrix(df_consumidor_filtrado)

    # # %%
    # # Analisa Missing Values para df_governo_filtrado
    # msno.matrix(df_governo_filtrado)

    # # %% [markdown]
    # # **Insights**:
    # # - Analizando os dados após filtragem, observa-se que a quantidade de valores nulos reduziram consideravelmente, logo iremos realizar a mescla de ambos os datasets para na sequência tratar os missing values restantes

    # # %%
    # #Checando possíveis inconsistências de mais de um separador em uma coluna, ignorando "substância"

    # def check_multiple_separators(df1, df2, sep=';'):
    #     dataframes = [df1, df2]
    #     issues = []

    #     for idx, df in enumerate(dataframes):
    #         for col in df.columns:  # Ignora a primeira coluna (SUBSTÂNCIA)
    #             if col != 'SUBSTÂNCIA':
    #                 for row_idx, value in enumerate(df[col]):
    #                     if isinstance(value, str) and sep in value:
    #                         issues.append(f'DataFrame {idx + 1}, Coluna "{col}", Linha {row_idx}: Valor com múltiplos separadores: {value}')
        
    #     if issues:
    #         print('Foram encontrados os seguintes problemas:')
    #         for issue in issues:
    #             print(issue)
    #     else:
    #         print('Nenhum problema encontrado nos DataFrames.')

    # # Carregue seus DataFrames aqui
    # df1 = df_consumidor_filtrado
    # df2 = df_governo_filtrado

    # check_multiple_separators(df1, df2)

    # # %%
    # def tratar_valores_multiplos_separadores(valor, sep=";"):
    #     if isinstance(valor, str):  # Verifica se o valor é uma string
    #         return sep.join(valor.split(sep, 1))
    #     else:
    #         return valor  # Retorna o valor original se não for uma string

    # # Aplicando a função de tratamento à coluna "APRESENTAÇÃO"
    # df_consumidor_filtrado["APRESENTAÇÃO"] = df_consumidor_filtrado["APRESENTAÇÃO"].apply(tratar_valores_multiplos_separadores)
    # df_governo_filtrado["APRESENTAÇÃO"] = df_governo_filtrado["APRESENTAÇÃO"].apply(tratar_valores_multiplos_separadores)

    # # %%
    # # Mescla dataframes
    # lista_join = ['CLASSE TERAPÊUTICA', 'REGIME DE PREÇO', 'LABORATÓRIO', 'LISTA DE CONCESSÃO DE CRÉDITO TRIBUTÁRIO (PIS/COFINS)', 'TARJA',
    #             'RESTRIÇÃO HOSPITALAR', 'ICMS 0%', 'TIPO DE PRODUTO (STATUS DO PRODUTO)', 'SUBSTÂNCIA','CNPJ', 'PRODUTO', 'CAP', 'CONFAZ 87',
    #             'PF Sem Impostos', 'REGISTRO', 'APRESENTAÇÃO', 'PF 0%']

    # df_conjunto = df_consumidor_filtrado.merge(df_governo_filtrado, how='inner', on=lista_join)

    # # %%
    # printmd("#### Comparação entre os datasets")
    # printmd(f"- Dataset de consumidor filtrado: ~**{df_consumidor_filtrado.shape[0]}** medicamentos, contendo **{df_consumidor_filtrado.shape[1]}** colunas")
    # printmd(f"- Dataset do governo filtrado: ~**{df_governo_filtrado.shape[0]}** medicamentos, contendo **{df_governo_filtrado.shape[1]}** colunas")
    # printmd(f"- Dataset do conjunto: ~**{df_conjunto.shape[0]}** medicamentos, contendo **{df_conjunto.shape[1]}** colunas")

    # # %% [markdown]
    # # **Insights**:
    # # - Observa-se que após mesclar os dados, perdeu-se apenas 5 registros, preservando a grande maioria dos dados.

    # # %%
    # # Verifica missing para df_conjunto
    # msno.matrix(df_conjunto)

    # # %%
    # df_conjunto.isnull().sum().sort_values(ascending=False)

    # # %% [markdown]
    # # **Insights**:
    # # - Como existem algumas entradas em que não temos os campos importantes como `PF Sem Impostos` e `PMVG Sem Impostos`, estas linhas serão removidas do dataset
    # # - Adicionalmente, criado coluna de Imposto PMC, o qual será utilizado para calcular os missing values em PMC 0%

    # # %%
    # # Remove linhas em que as colunas em que as colunas 'PF Sem Impostos', 'PMVG Sem Impostos' apresentam valores nulos.
    # df_conjunto.dropna(subset=['PF Sem Impostos', 'PMVG Sem Impostos'], inplace=True)
    # df_conjunto.isnull().sum().sort_values(ascending=False)

    # # %%
    # # Calcula coluna imposto_PMC com base no PMC 0% e PF 0%
    # df_conjunto['imposto_PMC'] = (df_conjunto['PMC 0%'] - df_conjunto['PF 0%'])/df_conjunto['PF 0%']*100

    # # %%
    # describe = df_conjunto['imposto_PMC'].describe()
    # describe['mode']= df_conjunto['imposto_PMC'].mode()[0]
    # describe.T

    # # %% [markdown]
    # # **Insights**:
    # # - A partir dos valores obtidos, nota-se que o imposto varia entre `33.33%` e `38.54%`, onde a moda gira em torno de `34.14%`, a qual será utilizada para preenchimento dos valores nulos no dataset, permitindo o calculo do PMC 0%

    # # %%
    # # Substitui Imposto PMC nulos pela moda e calcula o valor do PMC 0% com base no PF Sem Imposto e Imposto PMC
    # df_conjunto['imposto_PMC'].fillna(df_conjunto['imposto_PMC'].mode()[0], inplace=True)
    # df_conjunto['PMC 0%'].fillna(df_conjunto['PF 0%']*(df_conjunto['imposto_PMC']*100 + 1), inplace=True)
    # df_conjunto.drop(columns=['imposto_PMC'], inplace=True)

    # # %%
    # # Valida missing
    # df_conjunto.isnull().sum().sort_values(ascending=False)

    # # %% [markdown]
    # # **Insights**:
    # # - Uma vez tratado os missing values, importante verificar os dados das demais colunas

    # # %% [markdown]
    # # ### Tratamento de dados

    # # %% [markdown]
    # # **Insights**:
    # # - Após analisar as features do tipo Factor do dataframe, ajustado tipo no `df_conjunto`.

    # # %%
    # # Converte tipos de dados para os campos tipo Factor
    # factor_columns = ['CONFAZ 87', 'RESTRIÇÃO HOSPITALAR', 'TARJA',
    #     'SUBSTÂNCIA', 'REGIME DE PREÇO', 'ICMS 0%', 'CLASSE TERAPÊUTICA', 'CAP', 'LABORATÓRIO', 'PRODUTO', 'TIPO DE PRODUTO (STATUS DO PRODUTO)',
    #     'LISTA DE CONCESSÃO DE CRÉDITO TRIBUTÁRIO (PIS/COFINS)', ]
    # for col in factor_columns:
    #     df_conjunto[col] = df_conjunto[col].astype('category')

    # # %% [markdown]
    # # ## Análise exploratória dos dados

    # # %% [markdown]
    # # ### Custo dos medicamentos

    # # %%
    # # Prepara um dataframe apenas com as variáveis numéricas
    # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # dfNumeric = df_conjunto.select_dtypes(include=numerics)

    # # %%
    # # Estatísticas das variáveis numéricas
    # stats = dfNumeric.describe().T
    # stats['skew'] = dfNumeric.skew()
    # stats['mode'] = dfNumeric.mode().iloc[0]
    # stats['kurtosis'] = dfNumeric.kurtosis()
    # stats['iqr'] = stats['75%'] - stats['25%']
    # stats['variance'] = dfNumeric.var()
    # stats

    # # %% [markdown]
    # # **Insights**:
    # # - Analisando as estatísticas, observa-se que os valores dos produtos estão altamente concentrados nos valores mais baixos, com diversos outliers para os valores mais altos. 

    # # %%
    # # Top 5 medicamentos mais caros e suas respectivas classes terapeuticas
    # df_conjunto.sort_values(by='PF Sem Impostos', ascending=False).head(5)[['PRODUTO', 'CLASSE TERAPÊUTICA', 'PF Sem Impostos']]

    # # %%
    # # Top 5 medicamentos mais baratos e suas respectivas classes terapeuticas
    # df_conjunto.sort_values(by='PF Sem Impostos', ascending=True).head(5)[['PRODUTO', 'CLASSE TERAPÊUTICA', 'PF Sem Impostos']]

    # # %%
    # # Boxplot da distribuição dos preços dos medicamentos
    # sns.boxplot(data=df_conjunto, x='PF Sem Impostos')
    # plt.suptitle('Distribuição dos preços dos medicamentos', fontsize=16)
    # plt.show()

    # # %%
    # # Top 10 classes terapeuticas com maior quantidade de medicamentos
    # df_conjunto['CLASSE TERAPÊUTICA'].value_counts().head(10).plot(kind='barh')
    # plt.suptitle('Top 10 classes terapeuticas com maior quantidade de medicamentos', fontsize=16)

    # # %%
    # # Top 10 fabricantes com maior quantidade de medicamentos
    # df_conjunto['LABORATÓRIO'].value_counts().head(10).plot(kind='barh')
    # plt.suptitle('Top 10 fabricantes com maior quantidade de medicamentos', fontsize=16)

    # # %%
    # # Total de medicamentos por tipo de produto
    # df_conjunto['TIPO DE PRODUTO (STATUS DO PRODUTO)'].value_counts().plot(kind='barh')
    # plt.suptitle('Total de medicamentos por tipo de produto', fontsize=16)

    # # %%
    # # Total de medicamentos por Tarja
    # df_conjunto['TARJA'].value_counts().plot(kind='barh')
    # plt.suptitle('Total de medicamentos por Tarja', fontsize=16)

    # # %%
    # # Seleciona features categóricas 
    # dfCategorical = df_conjunto.select_dtypes(include=['object', 'category'])
    # dfCategorical.describe(include=['object', 'category']).T

    # # %% [markdown]
    # # ### Correlações entre variáveis numéricas

    # # %%
    # # Análise de correlação entre as variáveis
    # corr = dfNumeric.corr()
    # matrix = np.triu(corr)
    # plt.figure(figsize=(10,6))
    # ax = sns.heatmap(corr, mask=matrix, annot=True, fmt='.2f', square=True)
    # ax.set(xlabel="", ylabel="")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # plt.show()

    # # %% [markdown]
    # # **Insights**:
    # # - Conforme já sabido, uma vez que as variáveis numéricas são todas derivadas do `PF Sem Impostos`, todas as variaveis apresentam alta correlação entre si.

    # # %% [markdown]
    # # ## Análise multivariada

    # # %%
    # # Agrupar por "PRODUTO" e combinar "SUBSTÂNCIA"s relacionadas em uma lista
    # produtos_substancias = df_conjunto.groupby('PRODUTO')['SUBSTÂNCIA'].apply(list).reset_index()
    # produtos_substancias.head()

    # # %%
    # # Contar a frequência de ocorrência de cada substância
    # substancias_freq = df_conjunto['SUBSTÂNCIA'].value_counts().reset_index()
    # substancias_freq.columns = ['SUBSTÂNCIA', 'FREQUÊNCIA']

    # # %%
    # # Calcular o número de substâncias por produto
    # produtos_substancias['NUM_SUBSTÂNCIAS'] = produtos_substancias['SUBSTÂNCIA'].apply(lambda x: len(x))

    # # %%
    # produtos_substancias.head()

    # # %%
    # produtos_substancias['NUM_SUBSTÂNCIAS'].describe()

    # # %%
    # plt.hist(produtos_substancias['NUM_SUBSTÂNCIAS'], bins=20)
    # plt.xlabel('Número de Substâncias')
    # plt.ylabel('Frequência')
    # plt.title('Distribuição do Número de Substâncias por Produto')
    # plt.show()

    # # %%
    # sns.boxplot(x=produtos_substancias['NUM_SUBSTÂNCIAS'])
    # plt.xlabel('Número de Substâncias')
    # plt.title('Boxplot do Número de Substâncias por Produto')
    # plt.show()

    # # %% [markdown]
    # # **Insights**:
    # # - Embora a grande maioria dos medicamentos possuam apenas uma substancia, existem diversos produtos que são compostos da associação de múltiplos princípios ativos, conforme pode ser visto a partir da análise
    # # - A mesma informação pode ser confirmada a partir do boxplot apresentado

    # # %%
    # # Encontre as 10 substâncias mais frequentes
    # top_10_substancias = pd.DataFrame(df_conjunto['SUBSTÂNCIA'].value_counts().nlargest(10))
    # top_10_substancias

    # # %%
    # # Agrupando os dados por laboratório
    # laboratorio_groups = df_conjunto.groupby('LABORATÓRIO')

    # # Calculando as estatísticas dos preços de fábrica (PF) e dos preços ao consumidor (PMC) para cada laboratório
    # laboratorio_stats = laboratorio_groups.agg({
    #     'PF Sem Impostos': ['mean', 'median', 'min', 'max'],
    #     'PMC 0%': ['mean', 'median', 'min', 'max']
    # })

    # # Renomeando as colunas para facilitar a leitura
    # laboratorio_stats.columns = ['_'.join(col).strip() for col in laboratorio_stats.columns.values]

    # # Resetando o índice
    # laboratorio_stats.reset_index(inplace=True)

    # # Visualizando as estatísticas por laboratório
    # pd.DataFrame(laboratorio_stats.head())

    # # %%
    # # Identificando laboratórios com preços significativamente diferentes da média do mercado para essa substância
    # mean_pf = df_conjunto['PF Sem Impostos'].mean()
    # std_pf = df_conjunto['PF Sem Impostos'].std()

    # # Definindo um limiar arbitrário (2 desvios-padrão)
    # threshold = 2 * std_pf

    # # Encontrando laboratórios com preços médios significativamente diferentes da média do mercado
    # outliers = laboratorio_stats[(laboratorio_stats['PF Sem Impostos_mean'] < mean_pf - threshold) | (laboratorio_stats['PF Sem Impostos_mean'] > mean_pf + threshold)]
    # print("\nLaboratórios com preços significativamente diferentes da média do mercado:")
    # print(outliers)


    # # %% [markdown]
    # # **Insights**:
    # # - Laboratório `NOVARTIS BIOCIENCIAS S.A` pratica preços diferentes do mercado, porém análise mais detalhada pode ser importante para identificar o motivo.

    # # %%
    # #Aprofundarmento no laboratório NOVARTIS BIOCIENCIAS S.A 
    # novartis_produtos = df_conjunto[df_conjunto['LABORATÓRIO'] == 'NOVARTIS BIOCIENCIAS S.A']
    # novartis_produtos[['PRODUTO', 'SUBSTÂNCIA', 'PF Sem Impostos', 'PMC 0%']]





if __name__ == '__main__':
    main()