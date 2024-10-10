import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu

image_paths = ["assets/espacamento.png", "assets/gps.png", "assets/espacamento.png", "assets/nosso-time.png"]
cols = st.columns(len(image_paths))

image_path = "assets/logo.png"
image = Image.open(image_path)

model = joblib.load('modelofinal.joblib')

selected = option_menu(
        menu_title=None,
        options=["Página Principal", "Modelo Preditivo", "Como Utilizar"],
        icons=["compass", "rocket", "patch-question"],
        orientation="horizontal",
        styles={
            "nav-link": {
                "--hover-color":"#1e3350"
            },
            "nav-link-selected": {"background-color":"#e74c3c"}
        }
    )

if selected == "Página Principal":
    for image_path in image_paths:
        image = Image.open(image_path)
        st.image(image, use_column_width=True)

if selected == "Modelo Preditivo":
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.write("Dados carregados:")
            st.dataframe(df.head())

            # Limpeza e preparação dos dados
            df = df[df['meterSN'] != '>N<A']
            df['gain'].fillna(1, inplace=True)
            df['pulseCount'] = df['pulseCount'] * df['gain']
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['dateTimeSegundos'] = df['datetime'].astype(np.int64) // 10**9
            df['diffDateTime'] = df.groupby(['clientCode', 'meterSN']).dateTimeSegundos.diff()
            df['diffPulseCount'] = df.groupby(['clientCode', 'meterSN']).pulseCount.diff()
            df['diffPulseCountTempo'] = df['diffPulseCount'] / df['diffDateTime']
            df['diffDateTime'].fillna(0, inplace=True) 
            df['diffPulseCount'].fillna(0, inplace=True)
            df['diffPulseCountTempo'].fillna(0, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = df[['clientCode', 'clientIndex', 'meterSN', "pulseCount", 'diffPulseCount','datetime', 'diffDateTime', 'diffPulseCountTempo', 'dateTimeSegundos']]
            df['mediaCliente'] = df.groupby(['clientCode', 'meterSN']).diffPulseCount.transform('mean')
            df['desvioPadraoCliente'] = df.groupby(['clientCode', 'meterSN']).diffPulseCount.transform('std')
            df['mediaPCTCliente'] = df.groupby(['clientCode', 'meterSN']).diffPulseCountTempo.transform('mean')
            df['desvioPadraoPCTCliente'] = df.groupby(['clientCode', 'meterSN']).diffPulseCountTempo.transform('std')

            # Seleciona as variáveis para a predição
            X = df[['diffPulseCountTempo', 'mediaPCTCliente', 'desvioPadraoPCTCliente', 'diffDateTime']]

            # Prediz as categorias de anomalia
            y_pred = model.predict(X)
            
            # Adiciona as predições ao dataframe
            df['tipo_predito'] = y_pred
            df = df[['clientCode', 'clientIndex', 'meterSN', 'tipo_predito']]

            # Exibe os resultados da predição por categoria
            st.write("Resultados da predição por categoria de anomalia:")
            categoria_count = df['tipo_predito'].value_counts()
            st.write(categoria_count)

            # Filtra os clientes com e sem anomalias para o gráfico
            clientes_com_anomalia = df[df['tipo_predito'].isin(['cz', 'cn', 'sm1', 'sm7', 'sm30', 'dp3'])]['clientCode'].nunique()
            clientes_sem_anomalia = df[df['tipo_predito'] == 'c']['clientCode'].nunique()

            # Exibir tabelas para cada categoria com 'clientCode', 'clientIndex', 'meterSN', 'tipo_predito'
            st.write("Tabelas de clientes por categoria:")
            categorias_unicas = df['tipo_predito'].unique()

            for categoria in categorias_unicas:
                st.subheader(f"Categoria: {categoria}")
                tabela_categoria = df[df['tipo_predito'] == categoria][['clientCode', 'clientIndex', 'meterSN', 'tipo_predito']]
                st.dataframe(tabela_categoria)
                num_clientes_categoria = tabela_categoria['clientCode'].nunique()
                st.write(f"Número de clientes na categoria '{categoria}': {num_clientes_categoria}")
                
                num_medicoes_categoria = tabela_categoria[tabela_categoria['tipo_predito']==categoria].shape[0]
                st.write(f"Número de medições na categoria '{categoria}': {num_medicoes_categoria}")
        st.image(image, use_column_width=True)
    
if selected == "Como Utilizar":
    st.write("""
    Para que o modelo preditivo funcione corretamente, o arquivo CSV deve seguir os seguintes requisitos:

    1. **Colunas necessárias:**
        - `clientCode`: Identificador único de cada cliente.
        - `meterSN`: Número de série do medidor. Não deve conter valores iguais a '>N<A'.
        - `pulseCount`: Contagem de pulsos do medidor, que será ajustada pelo campo `gain`.
        - `gain`: Fator multiplicador aplicado à contagem de pulsos (`pulseCount`). Caso haja valores ausentes, o modelo os substituirá por 1.
        - `datetime`: Data e hora da leitura do medidor, no formato que permita a conversão para o tipo `datetime`.

    2. **Regras de tratamento de dados:**
        - O campo `gain` será preenchido automaticamente com o valor 1 caso contenha valores ausentes.
        - O modelo calculará automaticamente a diferença entre leituras consecutivas para criar as colunas `diffPulseCount` e `diffDateTime`, que são essenciais para o cálculo da taxa de variação do consumo (`diffPulseCountTempo`).

    3. **Outras considerações:**
        - O arquivo CSV deve estar bem formatado, sem valores nulos nas colunas principais, exceto onde o preenchimento é feito automaticamente (ex: `gain`).
        - O formato das datas deve ser legível e consistente para evitar erros na análise temporal.
        - Para otimizar os resultados, cada medidor deve ter múltiplas medições registradas ao longo do tempo, o que permitirá uma melhor avaliação das variações de consumo.

    Após carregar o arquivo CSV, o modelo processará os dados, identificará possíveis anomalias, e classificará os clientes nas diferentes categorias de análise.
    """)

    st.image(image, use_column_width=True)
