from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import resample
import pandas as pnd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

listas = ['month_2.csv', 'month_3.csv', 'month_4.csv', 'month_5.csv', 'month_6.csv']
df = []
for arquivo in listas:
    df += [pnd.read_csv(arquivo)]
#Concatena todos os dataframes em um único dataframe chamado df
df = pnd.concat(df)
#Chama o dataframe contido na variável chamada df
dadosCadastrais = pnd.read_csv('informacao_cadastral.csv')
usuariosUnicos = dadosCadastrais[dadosCadastrais.situacao == 'CONSUMINDO GÁS']['clientCode'].unique() 
#Organiza os dados dos usuários filtrados pela data
mesFiltrado = df[df['clientCode'].isin(usuariosUnicos)].sort_values(by='datetime') 
#Filtra meterSN diferente de '>N<A'
df = mesFiltrado[mesFiltrado['meterSN'] != '>N<A']
#Garante que todas as linhas com gain nulo sejam preenchidas com 1. Não é garantido que é o valor correto, mas é o melhor que podemos fazer
df['gain'].fillna(1, inplace=True)
#Corrige os pulsos para m²
df['pulseCount'] = df['pulseCount'] * df['gain']
#Cria a variação do pulseCount como uma coluna nova, calculando por grupo a diferença
df['datetime'] = pnd.to_datetime(df['datetime'])
df['dateTimeSegundos'] = df['datetime'].astype(np.int64) // 10**9
df['diffDateTime'] = df.groupby(['clientCode', 'meterSN']).dateTimeSegundos.diff()
df['diffPulseCount'] = df.groupby(['clientCode', 'meterSN']).pulseCount.diff()
df['diffPulseCountTempo'] = df['diffPulseCount'] / df['diffDateTime']
#Preenche os valores nulos (iniciais) com 0
df['diffDateTime'].fillna(0, inplace=True) 
df['diffPulseCount'].fillna(0, inplace=True)
df['diffPulseCountTempo'].fillna(0, inplace=True)
#Reseta o index
df.reset_index(drop=True, inplace=True)
#Seleciona as colunas que serão usadas
df = df[['clientCode', 'meterSN', "pulseCount", 'diffPulseCount','datetime', 'diffDateTime', 'diffPulseCountTempo', 'dateTimeSegundos']]
#Calcula a média e o desvio padrão do diffPulseCount por cliente
df['mediaCliente'] = df.groupby(['clientCode', 'meterSN']).diffPulseCount.transform('mean')
df['desvioPadraoCliente'] = df.groupby(['clientCode', 'meterSN']).diffPulseCount.transform('std')
df['diffDateTime'].describe()
df['mediaPCTCliente'] = df.groupby(['clientCode', 'meterSN']).diffPulseCountTempo.transform('mean')
df['desvioPadraoPCTCliente'] = df.groupby(['clientCode', 'meterSN']).diffPulseCountTempo.transform('std')

df['tipo'] = "c"

#Classifica os inidivíduos sem medições por longos períodos de tempo
df.loc[df['diffDateTime'] > 86400, 'tipo'] = "sm1"
df.loc[df['diffDateTime'] > 604800, 'tipo'] = "sm7"
df.loc[df['diffDateTime'] > 2592000, 'tipo'] = "sm30"

#Classifica consumo acima de 3 desvio padrão, consumo negativo e consumo zerado
df.loc[df['diffPulseCountTempo'] > df['mediaPCTCliente'] + 3 * df['desvioPadraoPCTCliente'], 'tipo'] = "dp3"
df.loc[df['diffPulseCountTempo'] < 0, 'tipo'] = "cn"
df.loc[(df['pulseCount'] == 0) & (df['diffPulseCountTempo'] < 0), 'tipo'] = "cz"

X = df[['diffPulseCountTempo', 'mediaPCTCliente', 'desvioPadraoPCTCliente', 'diffDateTime']]  # Features
y = df['tipo']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_majority = X_test[y_test == 'c']
X_minority = X_test[y_test != 'c']
y_majority = y_test[y_test == 'c']
y_minority = y_test[y_test != 'c']
X_majority_downsampled, y_majority_downsampled = resample(X_majority, y_majority, replace=False, n_samples=len(X_minority), random_state=42)
X_test_balanced = pnd.concat([X_majority_downsampled, X_minority])
y_test_balanced = pnd.concat([y_majority_downsampled, y_minority])
# Definindo o modelo
model = RandomForestClassifier(
    random_state=42,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=100,
    n_jobs=-1
)

print("Prestes a fazer .fit do modelo")
# Rodando o Grid Search no conjunto de treino
model.fit(X_train, y_train)

print("Prestes a fazer .predict do modelo")
# Avaliando o modelo no conjunto de teste
y_pred = model.predict(X_test_balanced)

print("O modelo rodou")
# Avaliar a acurácia
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_balanced, y_pred)
print(f"Accuracy on test data: {accuracy:.2f}")

print(classification_report(y_test_balanced, y_pred))