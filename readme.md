# 📊 Streamlit: Predição de Anomalias

Este projeto foi desenvolvido como parte de uma análise preditiva para identificar anomalias no consumo de gás em clientes de uma distribuidora. Utilizando **Python**, **Streamlit**, e diversas bibliotecas de manipulação e visualização de dados, o objetivo principal é prever categorias de anomalias em medições de consumo com base em características específicas extraídas dos dados.

## 🚀 Funcionalidades

- **Upload de CSV**: Faça upload de um arquivo no formato `.csv` contendo as informações dos clientes e medições.
- **Limpeza e Preparação de Dados**: O modelo automaticamente trata dados faltantes, normaliza colunas e faz transformações necessárias para o processo preditivo.
- **Predição de Anomalias**: Após o upload, o modelo preditivo processa os dados e classifica cada cliente em uma categoria de anomalia ou sem anomalia.
- **Visualização dos Resultados**: O aplicativo exibe os resultados das predições em formato de tabela, além de gráficos que mostram a distribuição de clientes com e sem anomalias.
- **Gráfico de Pizza**: Visualize a proporção de clientes com e sem anomalias por meio de um gráfico interativo.

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem principal do projeto.
- **Pandas & NumPy**: Manipulação e tratamento de dados.
- **Streamlit**: Framework para criação de aplicações web de forma rápida e eficiente.
- **Matplotlib**: Biblioteca de visualização para gerar gráficos.
- **Joblib**: Utilizado para carregar o modelo preditivo treinado.

## 📄 Como Utilizar

1. Clone o repositório:
    ```bash
    git clone https://github.com/souzajv/Streamlit-Grupo4.git
    ```
2. Crie um ambiente virtual, e incie

    Windows:
    ```bash
    python -m venv venv 
    ```
    ```bash
    venv\Scripts\activate
    ```
    Linux e MAC:
    ```bash
    python3 -m venv venv
    ```
    ```bash
    source venv/bin/activate
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
4. Execute o código:
    ```bash
    streamlit run app.py
    ```

## 💡 Motivação
Este projeto foi desenvolvido no contexto do meu estudo de predição de anomalias para uma distribuidora de gás. O objetivo era aplicar técnicas de ciência de dados para monitorar o comportamento do consumo e identificar inconsistências ou padrões incomuns que possam sinalizar anomalias.