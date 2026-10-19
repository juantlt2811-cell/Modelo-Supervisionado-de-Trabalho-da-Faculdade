import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

dados_teste = pd.read_csv('02_dados_tratados/dados_clusterizados_rotulados.csv')


x = dados_teste[["idade","renda_mensal","compras_mes","ticket_medio","engajamento_fidelidade"]]
y = dados_teste["rotulo_grupo"]


x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

padronizador = StandardScaler()

x_treino_padronizado = padronizador.fit_transform(x_treino)

x_teste_padronizado = padronizador.transform(x_teste)


def avaliar_modelo(modelo, x_teste, y_teste, nome_modelo="Modelo"):
    y_pred = modelo.predict(x_teste)
    
    print(f"\n--- Relatório de Classificação: {nome_modelo} ---")
    print(classification_report(y_teste, y_pred))

    matriz = confusion_matrix(y_teste, y_pred)
    grafico = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=modelo.classes_)
    

    grafico.plot(cmap='Blues', values_format='d')
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.xticks(rotation=45)
    plt.show()

modelo50 = MLPClassifier(
    hidden_layer_sizes = (5,),
    max_iter = 50,
    tol = 0,
    random_state = 42,
    verbose = True
)

modelo50.fit(x_treino_padronizado, y_treino)

y_pred_treino = modelo50.predict(x_treino_padronizado)
acuraciaTreino = accuracy_score(y_treino, y_pred_treino)
print(f"Acurácia de treino do Modelo: {acuraciaTreino: .2%}")

y_pred_teste = modelo50.predict(x_teste_padronizado)
acuracia = accuracy_score(y_teste, y_pred_teste)
print(f"Acurácia de teste do Modelo: {acuracia: .2%}")

print(f"\n--- Detalhes do Treinamento ---")
print(f"Quantidade real de iterações: {modelo50.n_iter_}")
print(f"Perda final (loss): {modelo50.loss_:.6f}")

plt.plot(modelo50.loss_curve_)
plt.xlabel("Iterações")
plt.ylabel("Loss")
plt.title("Curva de Loss")
plt.show()

avaliar_modelo(modelo50, x_teste_padronizado, y_teste, nome_modelo="MLP 50 Iterações")

modelo500 = MLPClassifier(
    hidden_layer_sizes = (5,),
    max_iter = 500,
    tol = 0,
    random_state = 42,
    verbose = True
)

modelo500.fit(x_treino_padronizado, y_treino)

y_pred_treino = modelo500.predict(x_treino_padronizado)
acuraciaTreino = accuracy_score(y_treino, y_pred_treino)
print(f"Acurácia de treino do Modelo: {acuraciaTreino: .2%}")

y_pred_teste = modelo500.predict(x_teste_padronizado)
acuracia = accuracy_score(y_teste, y_pred_teste)
print(f"Acurácia de teste do Modelo: {acuracia: .2%}")


print(f"\n--- Detalhes do Treinamento ---")
print(f"Quantidade real de iterações: {modelo500.n_iter_}")
print(f"Perda final (loss): {modelo500.loss_:.6f}")

plt.plot(modelo500.loss_curve_)
plt.xlabel("Iterações")
plt.ylabel("Loss")
plt.title("Curva de Loss")
plt.show()

avaliar_modelo(modelo500, x_teste_padronizado, y_teste, nome_modelo="MLP 500 Iterações")




modelo5000 = MLPClassifier(
    hidden_layer_sizes = (5,),
    max_iter = 5000,
    tol = 0,
    random_state = 42,
    verbose = True
)

modelo5000.fit(x_treino_padronizado, y_treino)

y_pred_treino = modelo5000.predict(x_treino_padronizado)
acuraciaTreino = accuracy_score(y_treino, y_pred_treino)
print(f"Acurácia de treino do Modelo: {acuraciaTreino: .2%}")

y_pred_teste = modelo5000.predict(x_teste_padronizado)
acuracia = accuracy_score(y_teste, y_pred_teste)
print(f"Acurácia de teste do Modelo: {acuracia: .2%}")

print(f"\n--- Detalhes do Treinamento ---")
print(f"Quantidade real de iterações: {modelo5000.n_iter_}")
print(f"Perda final (loss): {modelo5000.loss_:.6f}")

plt.plot(modelo5000.loss_curve_)
plt.xlabel("Iterações")
plt.ylabel("Loss")
plt.title("Curva de Loss")
plt.show()

avaliar_modelo(modelo5000, x_teste_padronizado, y_teste, nome_modelo="MLP 5000 Iterações")

novos_compradores = [
    {"idade": 40, "renda_mensal": 3001.55, "compras_mes": 4, "ticket_medio": 5.0, "engajamento_fidelidade": 30.3}, # Deve retornar perfil_baixa_frequencia
    {"idade": 46, "renda_mensal": 6809.31, "compras_mes": 9, "ticket_medio": 171.51, "engajamento_fidelidade": 70.3}, # Deve retornar perfil_equilibrado
    {"idade": 30, "renda_mensal": 3000.00, "compras_mes": 5, "ticket_medio": 500.00, "engajamento_fidelidade": 20.0}, # Deve retornar perfil_equilibrado
    {"idade": 70, "renda_mensal": 3000.00, "compras_mes": 7, "ticket_medio": 500.00, "engajamento_fidelidade": 50.0}, # Deve retornar perfil_alto_consumo
    {"idade": 90, "renda_mensal": 3000.00, "compras_mes": 7, "ticket_medio": 600.00, "engajamento_fidelidade": 10.0} # Deve retornar perfil_alto_consumo
]


df_novos_compradores = pd.DataFrame(novos_compradores)


compradores_padronizados = padronizador.transform(df_novos_compradores)


predicoes = modelo500.predict(compradores_padronizados)

print("Testes com o modelo escolhido: Modelo de 500 iterações")
for indice, predicao in enumerate(predicoes, start=1):
    print(f"Grupo/Rótulo predito para o comprador {indice}: {predicao}")