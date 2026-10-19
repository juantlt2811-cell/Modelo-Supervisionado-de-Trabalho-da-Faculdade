# Grupo 06 - Varejo e Compras

1. Dados Recebidos
2. Dados Tratados
3. Códigos
4. Resultados

### 1 - O que são os dados que nós recebemos

Nosso grupo recebeu o desafio de desenvolver uma rede neural capaz de avaliar qual é o perfil daquele comprador (Alto consumo, Baixo consumo ou Equilibrado).

A base de dados possui os atributos de idade, renda_mensal, compras_mes, ticket_medio e engajamento_fidelidade.

Para a nossa análise, foi pedido que usassemos as variáveis compras_mes no eixo X e ticket_medio no eixo Y, para que assim pudessemos avaliar e separar os perfis em grupos parecidos.

Observação:

- Os dados são sintéticos e destinam-se exclusivamente à atividade acadêmica.
- Não crie conclusões causais; os dados permitem identificar padrões, não provar relações de causa e efeito.
- Não transforme os resultados em decisões reais sobre pessoas, crédito, saúde, trabalho, educação, ambiente ou qualquer outro contexto real.

### 2 - Tratamento dos dados

Para começar o tratamento dos dados, utilizamos o K-Means para ver onde tinha a quebra no método do cotovelo e qual quantidade de grupos estava mais alinhada ao propósito do método de silhueta.

Optamos por atribuir ao campo K_ESCOLHIDO o número 3 por se encaixar exatamente nessa métrica que dissemos.

| k   | inercia            | silhueta            |
| --- | ------------------ | ------------------- |
| 2   | 17340.468207769838 | 0.5012076354846243  |
| 3   | 11368.850317818078 | 0.37945870988498537 |
| 4   | 10237.173886958482 | 0.27950449243484454 |
| 5   | 9481.148602353034  | 0.2664826086678936  |

Com os dados clusterizados pelo K-means, fizemos uma rápida automação para que o python organizasse os dados da seguinte forma:

Grupo 0 - perfil_baixa_frequencia
Grupo 1 - perfil_alto_consumo
Grupo 2 - perfil_equilibrado

Código utilizado para automação:

```python
import pandas as pd

dados_teste = pd.read_csv('02_dados_tratados/dados_clusterizados_rotulados.csv')

mapeamento = {
    1: "perfil_alto_consumo",
    2: "perfil_equilibrado"
}

dados_teste['rotulo_grupo'] = dados_teste['cluster_kmeans'].map(mapeamento).fillna("perfil_baixa_frequencia")

dados_teste.to_csv('02_dados_tratados/dados_clusterizados_rotulados.csv', index=False)

print(dados_teste)
```

### 3 - Como funciona nossa rede neural?

Imports de bibliotecas: Utilizou-se a biblioteca pandas para a leitura e o tratamento dos dados; a scikit-learn para a criação, processamento e treinamento dos modelos; e a matplotlib para a geração dos gráficos explicativos.

```python
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
```

Separação de Dados: Inicialmente, o arquivo dados_clusterizados_rotulados.csv é lido. Em seguida, esses dados são extraídos e armazenados nas variáveis X e Y. A variável X guarda as características dos clientes ("idade", "renda_mensal", "compras_mes", "ticket_medio" e "engajamento_fidelidade"), enquanto a variável Y armazena a resposta, que é o "rotulo_grupo". Logo abaixo, é feita a divisão entre dados de treino e teste gerando as variáveis x_treino, x_teste, y_treino e y_teste. Nessa etapa, definiu-se que 20% dos dados serão reservados para o teste e os 80% restantes serão utilizados para o treino do modelo.

Padronização: O padronizador serve para normalizar a escala dos dados, deixando-os com média 0 e desvio padrão 1. Ao executar padronizador.fit_transform(x_treino), o algoritmo analisa as colunas de treino para descobrir sua média e desvio padrão e, em seguida, ajusta esses valores. Na linha seguinte, padronizador.transform(x_teste), garante-se que os dados de teste sejam transformados usando exatamente a mesma média e desvio padrão aprendidos no treino, evitando que informações do teste vazem para o modelo antes da hora.

Modelo 1: 50 Iterações
O gráfico da função de custo (loss) revelou uma queda gradual e contínua ao longo do tempo. Como o limite de épocas era muito restrito, a rede não teve tempo suficiente para convergir, e o treinamento foi encerrado antes que o erro pudesse se estabilizar.

Modelo 2: 500 Iterações
Com uma janela de treinamento maior, o comportamento da rede mudou drasticamente. Observamos uma queda brusca do erro próxima à 100ª iteração, seguida por um declínio estabilizado em linha reta. Perto do fim do processo — mais especificamente a partir da iteração 480 —, a capacidade de aprendizado estagnou. O erro permaneceu travado em 0,79, conseguindo uma redução mínima para 0,78 apenas na última iteração.

Modelo 3: 5000 Iterações
Apesar do limite amplo, o treinamento não alcançou as 5000 iterações, parando precocemente na iteração 890. Isso ocorreu devido a um segundo mecanismo de segurança silencioso do Scikit-Learn: o parâmetro n_iter_no_change (que possui valor 10 por padrão). Mesmo com o teto estendido e a tolerância forçada a zero, a biblioteca monitora o platô de aprendizado. Ao perceber que a rede não conseguia mais otimizar o erro por 10 iterações consecutivas, o algoritmo interrompeu o processo automaticamente para evitar desperdício de processamento computacional. O uso do n_iter_no change = 5000 faria com que ele não aprendesse, ele estaria "decorando" os dados.

### 4 - Os resultados apresentados

Avaliando os três modelos, percebemos que o número de iterações influenciou positivamente, aumentando gradualmente a acurácia de treino. Apesar disso, percebe-se que esse aumento de 500 iterações para 890 não possui uma evolução tão drástica, o que nos leva a considerar que o custo de manter um modelo de 5000 iterações não se faz tão necessário.

Falando sobre a acurácia de teste, nota-se que o experimento B obteve uma precisão maior do que a do experimento C (0.13% a mais). Pode parecer uma diferença mínima, mas isso mostra que ao longo das iterações o modelo "decorou" os dados e aprendeu menos.

| Experimento | max_iter | Iterações executadas | Perda final | Acurácia treino | Acurácia teste | Observação                                                                                                       |
| ----------- | -------- | -------------------- | ----------- | --------------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| A           | 50       | 50                   | 0.082434    | 99.22%          | 99.19%         | Baixo custo de processamento, menor acurácia dos três                                                            |
| B           | 500      | 500                  | 0.007805    | 99.88%          | 99.94%         | Com a tolerância 0, o modelo alcançou as 500 iterações e obteve uma acurácia de teste extremamente alta          |
| C           | 5000     | 890                  | 0.00507546  | 99.94%          | 99.81%         | Nem sempre a maior quantidade de iterações é a ideal, pois o modelo pode "decorar" os dados ao invés de aprender |

Com essas avaliações, optamos por definir o experimento B como o melhor dos três casos. O seu custo relativamente baixo somado com o seu alto desempenho e acurácia alta, faz dele o modelo mais bem preparado para conseguir prever que tipo de perfil o comprador terá.

Como parte da comprovação, utilizamos um código para avaliar a matriz de confusão por meio de gráficos

```python
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
```

A função acima recebe o modelo e os dados de teste como parametro para realizar os gráficos da matriz de confusão de acordo com o modelo que a gente quer. O parametro nome_modelo é opcional com um valor padrão para utilizar na hora de criar os gráficos, mas nós alteramos de acordo com os modelos.

O modelo de 50 iterações foi o que mais errou, principalmente na hora de definir os compradores de perfil_equilibrado. Como o gráfico de dispersão mostra, temos alguns compradores com perfil_baixa_frequencia que acabam se encaixando próximos ao perfil_equilibrado, assim como alguns compradores de perfil_alto_consumo acabam tendo a mesma consequência.

Já no modelo de 5000 iterações, a confusão foi bem menor. O experimento C confundiu apenas 3 compradores que deveriam ser de perfil_baixa_frequencia, colocando eles como perfil_equilibrado.

O nosso modelo de 500 iterações obteve o melhor resultado. Enquanto 5000 iterações nos trouxeram 3 compradores com seus perfis errados, o nosso experimento B errou apenas 1 deles. Isso não é apenas um número pequeno, mas sim um ponto forte do porquê esse é o melhor modelo. Embora os números sejam bem próximos, a acurácia do experimento B - tanto em treino, quanto em teste - somada com uma matriz de confusão quase nula e o aprendizado real do modelo, sem ficar decorando o treino, comprovam a nossa escolha.

Para fechar as análises, testamos o modelo com alguns dados que já existem na tabela e outros completamente novos

```python

novos_compradores = [
    {"idade": 40, "renda_mensal": 3001.55, "compras_mes": 4, "ticket_medio": 5.0, "engajamento_fidelidade": 30.3}, # Deve retornar perfil_baixa_frequencia -> pertence a base de dados
    {"idade": 46, "renda_mensal": 6809.31, "compras_mes": 9, "ticket_medio": 171.51, "engajamento_fidelidade": 70.3}, # Deve retornar perfil_equilibrado -> pertence a base de dados
    {"idade": 30, "renda_mensal": 3000.00, "compras_mes": 5, "ticket_medio": 500.00, "engajamento_fidelidade": 20.0}, # Deve retornar perfil_equilibrado
    {"idade": 70, "renda_mensal": 3000.00, "compras_mes": 7, "ticket_medio": 500.00, "engajamento_fidelidade": 50.0}, # Deve retornar perfil_alto_consumo
    {"idade": 90, "renda_mensal": 3000.00, "compras_mes": 7, "ticket_medio": 600.00, "engajamento_fidelidade": 10.0} # Deve retornar perfil_alto_consumo
]


df_novos_compradores = pd.DataFrame(novos_compradores)


compradores_padronizados = padronizador.transform(df_novos_compradores)


predicoes = modelo500.predict(compradores_padronizados)


for indice, predicao in enumerate(predicoes, start=1):
    print(f"Grupo/Rótulo predito para o comprador {indice}: {predicao}")
```

Retorno esperado:
Grupo/Rótulo predito para o comprador 1: perfil_baixa_frequencia
Grupo/Rótulo predito para o comprador 2: perfil_equilibrado
Grupo/Rótulo predito para o comprador 3: perfil_equilibrado
Grupo/Rótulo predito para o comprador 4: perfil_alto_consumo
Grupo/Rótulo predito para o comprador 5: perfil_alto_consumo
