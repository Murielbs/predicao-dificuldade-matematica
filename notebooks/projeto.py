"""
PROJETO-DESAFIO – INTELIGÊNCIA ARTIFICIAL NA PRÁTICA
Objetivo: Predizer dificuldade em matemática (nota < 10) usando Machine Learning.
Dataset: Student Performance (UCI Machine Learning Repository)
"""

# 1. BIBLIOTECAS

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


# 2. CARREGAR DADOS

print("\n=== CARREGANDO DADOS ===\n")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
df = pd.read_csv(url, sep=";")


# 3. ANÁLISE EXPLORATÓRIA

print("\n=== ANÁLISE INICIAL ===\n")
print("5 primeiras linhas:\n", df.head())
print("\nEstatísticas descritivas:\n", df.describe())

# Criar variável alvo (dificuldade = 1 se nota G3 < 10)
df['math_difficulty'] = df['G3'].apply(lambda x: 1 if x < 10 else 0)

# Gráfico de distribuição
plt.figure(figsize=(8, 4))
df['math_difficulty'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribuição de Dificuldade em Matemática (0 = Sem dificuldade, 1 = Com dificuldade)')
plt.savefig('distribuicao_dificuldade.png')
print("\nGráfico salvo como 'distribuicao_dificuldade.png'")


# 4. PRÉ-PROCESSAMENTO

print("\n=== PRÉ-PROCESSAMENTO ===\n")

# Selecionar 10 variáveis conforme o desafio
features = ['sex', 'age', 'Medu', 'Fedu', 'studytime', 'failures', 'absences', 'G1', 'G2', 'goout']
X = df[features]
y = df['math_difficulty']

# Codificar variáveis categóricas (sex: F=0, M=1)
X['sex'] = LabelEncoder().fit_transform(X['sex'])

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir dados (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 5. TREINAMENTO

print("\n=== TREINAMENTO ===\n")

# Modelo 1: Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Modelo 2: Regressão Logística
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)


# 6. AVALIAÇÃO

print("\n=== RESULTADOS ===\n")

# Previsões
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)

# Métricas
print("Random Forest:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.2%}")
print(classification_report(y_test, y_pred_rf))

print("\nRegressão Logística:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_lr):.2%}")
print(classification_report(y_test, y_pred_lr))


# 7. SALVAR MELHOR MODEL
melhor_modelo = rf if accuracy_score(y_test, y_pred_rf) > accuracy_score(y_test, y_pred_lr) else lr

with open('melhor_modelo.pkl', 'wb') as f:
    pickle.dump(melhor_modelo, f)

print(f"\nModelo salvo: {'Random Forest' if melhor_modelo == rf else 'Regressão Logística'}")


