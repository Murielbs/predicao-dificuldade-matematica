# Desafio IA - Previsão de Dificuldade em Matemática

## 1. Importação das Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import gradio as gr

## 2. Carregamento dos Dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
data = pd.read_csv(url, sep=';')
print("Dimensões do dataset:", data.shape)
data.head()

## 3. Análise Exploratória
# Distribuição das notas finais
plt.figure(figsize=(8,5))
sns.histplot(data['G3'], bins=10, kde=True)
plt.title('Distribuição das Notas Finais de Matemática')
plt.xlabel('Nota Final')
plt.ylabel('Frequência')
plt.show()

# Tempo de estudo vs Nota
plt.figure(figsize=(8,5))
sns.boxplot(x='studytime', y='G3', data=data)
plt.title('Tempo de Estudo x Nota Final')
plt.xlabel('Tempo de Estudo')
plt.ylabel('Nota Final')
plt.show()

# Apoio escolar
plt.figure(figsize=(6,4))
sns.countplot(x='schoolsup', data=data)
plt.title('Distribuição de Apoio Escolar')
plt.show()

## 4. Pré-processamento dos Dados
# Criação da variável alvo
data['dificuldade'] = data['G3'].apply(lambda x: 1 if x <= 10 else 0)

# Remoção das notas anteriores (evita vazamento de dados)
data = data.drop(['G1', 'G2', 'G3'], axis=1)

# Seleção de variáveis úteis
cols = ['sex', 'age', 'studytime', 'failures', 'schoolsup', 'famsup', 'absences', 'dificuldade']
data = data[cols]

# Codificação de variáveis categóricas
le = LabelEncoder()
for col in ['sex', 'schoolsup', 'famsup']:
    data[col] = le.fit_transform(data[col])

## 5. Divisão dos Dados
dados = data.copy()
X = dados.drop('dificuldade', axis=1)
y = dados['dificuldade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Tamanho do treino:", X_train.shape)
print("Tamanho do teste:", X_test.shape)

## 6. Modelagem
# Modelo 1: Árvore de Decisão
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
acc_dtc = accuracy_score(y_test, y_pred_dtc)
print("Acurácia - Árvore de Decisão:", acc_dtc)
print(classification_report(y_test, y_pred_dtc))
sns.heatmap(confusion_matrix(y_test, y_pred_dtc), annot=True, fmt='d')
plt.title('Matriz de Confusão - Árvore de Decisão')
plt.show()

# Modelo 2: KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("Acurácia - KNN:", acc_knn)
print(classification_report(y_test, y_pred_knn))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d')
plt.title('Matriz de Confusão - KNN')
plt.show()

# Comparação
print("Melhor modelo:", "Árvore de Decisão" if acc_dtc > acc_knn else "KNN")

## 7. Nível II - 30 Execuções para Avaliação
from tqdm import tqdm
results_dtc, results_knn = [], []

for _ in tqdm(range(30)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dtc.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    results_dtc.append(accuracy_score(y_test, dtc.predict(X_test)))
    results_knn.append(accuracy_score(y_test, knn.predict(X_test)))

print("Média DTC:", np.mean(results_dtc), "/ Desvio padrão:", np.std(results_dtc))
print("Média KNN:", np.mean(results_knn), "/ Desvio padrão:", np.std(results_knn))

## 8. Salvando o Melhor Modelo (Pickle)
melhor_modelo = dtc if np.mean(results_dtc) > np.mean(results_knn) else knn
with open("../models/modelo_final.pkl", "wb") as f:
    pickle.dump(melhor_modelo, f)

## 9. Interface com Gradio
def prever_dificuldade(sex, age, studytime, failures, schoolsup, famsup, absences):
    entrada = pd.DataFrame({
        'sex': [0 if sex == 'Feminino' else 1],
        'age': [age],
        'studytime': [studytime],
        'failures': [failures],
        'schoolsup': [1 if schoolsup == 'Sim' else 0],
        'famsup': [1 if famsup == 'Sim' else 0],
        'absences': [absences]
    })
    with open("../models/modelo_final.pkl", "rb") as f:
        modelo = pickle.load(f)
    pred = modelo.predict(entrada)[0]
    return "Com dificuldade" if pred == 1 else "Sem dificuldade"

iface = gr.Interface(
    fn=prever_dificuldade,
    inputs=[
        gr.Radio(['Masculino', 'Feminino'], label="Sexo"),
        gr.Slider(15, 22, label="Idade"),
        gr.Slider(1, 4, label="Tempo de Estudo (1-4)"),
        gr.Slider(0, 3, label="Reprovações"),
        gr.Radio(['Sim', 'Não'], label="Apoio Escolar"),
        gr.Radio(['Sim', 'Não'], label="Apoio Familiar"),
        gr.Slider(0, 100, label="Faltas")
    ],
    outputs="text",
    title="Preditor de Dificuldade em Matemática"
)

# Para ativar a interface, descomente a linha abaixo:
# iface.launch()
