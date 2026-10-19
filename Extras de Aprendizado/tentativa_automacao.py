import pandas as pd

# Carregar o arquivo CSV
dados_teste = pd.read_csv('02_dados_tratados/dados_clusterizados_rotulados.csv')

# Definir o dicionário de mapeamento
mapeamento = {
    1: "perfil_alto_consumo",
    2: "perfil_equilibrado"
}

# Criar a coluna mapeando os valores e preenchendo o restante com o "else"
dados_teste['rotulo_grupo'] = dados_teste['cluster_kmeans'].map(mapeamento).fillna("perfil_baixa_frequencia")

# Salvar o DataFrame atualizado de volta no mesmo arquivo CSV
dados_teste.to_csv('02_dados_tratados/dados_clusterizados_rotulados.csv', index=False)

print(dados_teste)