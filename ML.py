import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split


# Ler o arquivo CSV
caminho = 'C:\\TCC\\bases_tempo\\'
df = pd.read_csv(caminho + 'dados_finais.csv', sep=',', dayfirst=True, parse_dates=['Data'], decimal=',')

# Convertendo a coluna 'Data' para o formato datetime
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')

# Criando colunas para o mês e o ano
df['Mes'] = df['Data'].dt.month
df['Ano'] = df['Data'].dt.year

# Convertendo as colunas 'Temp Comp Media' e 'Precipitacao' para numérico
df['Temp Comp Media'] = pd.to_numeric(df['Temp Comp Media'], errors='coerce')
df['Precipitacao'] = pd.to_numeric(df['Precipitacao'], errors='coerce').replace(0, 1)  # Substituir 0 por 1

# Agrupando os dados por estação, mês e ano
df_agrupado = df.groupby(['Estacao', 'Mes', 'Ano']).agg({'Precipitacao':'sum', 'Temp Comp Media':'mean'}).reset_index()

# Calculando as variações percentuais anuais por estação
df_agrupado['Variação Anual Estação Temp Comp Media'] = df_agrupado.groupby(['Estacao', 'Mes'])['Temp Comp Media'].pct_change(fill_method=None).round(3)
df_agrupado['Variação Anual Estação Precipitacao'] = df_agrupado.groupby(['Estacao', 'Mes'])['Precipitacao'].pct_change(fill_method=None).round(3)

# Agrupando os dados por mês e ano para calcular as variações totais
df_total = df.groupby(['Mes', 'Ano']).agg({'Precipitacao':'sum', 'Temp Comp Media':'mean'}).reset_index()

# Calculando as variações percentuais anuais totais
df_total['Variação Total Anual Temp Comp Media'] = df_total.groupby('Mes')['Temp Comp Media'].pct_change(fill_method=None).round(3)
df_total['Variação Total Anual Precipitacao'] = df_total.groupby('Mes')['Precipitacao'].pct_change(fill_method=None).round(3)

# Juntando os dataframes
df_final = pd.merge(df_agrupado, df_total[['Mes', 'Ano', 'Variação Total Anual Temp Comp Media', 'Variação Total Anual Precipitacao']], on=['Mes', 'Ano'])

# Ler o arquivo CSV do ENSO
dfenso = pd.read_csv(caminho + 'ENSO.csv')

# Convertendo a coluna 'Date' para o formato datetime
dfenso['Date'] = pd.to_datetime(dfenso['Date'])

# Criando colunas para o mês e o ano
dfenso['Mes'] = dfenso['Date'].dt.month
dfenso['Ano'] = dfenso['Date'].dt.year

# Selecionando apenas as colunas 'Mes', 'Ano' e 'ONI'
dfenso = dfenso[['Mes', 'Ano', 'ONI']]

# Criar a coluna ONI ajustado com ONI + 20
dfenso['ONI_ajustado'] = dfenso['ONI'] + 20

# Calcular a variação mensal dos valores ONI ajustado por mês e ano
dfenso['Variação_ONI'] = dfenso.groupby('Mes')['ONI_ajustado'].pct_change().round(3)

# Substituir infinito e menos infinito por NaN
dfenso['Variação_ONI'] = dfenso['Variação_ONI'].replace([np.inf, -np.inf], np.nan)

# Preencher NaN com a diferença entre o valor atual e o próximo valor não-zero
dfenso['Variação_ONI'] = dfenso['Variação_ONI'].fillna(dfenso.groupby('Mes')['ONI'].diff())

# Verificar se existem dados faltantes
if dfenso.isnull().any().any():
    print("Existem dados faltantes no DataFrame do ENSO.")
else:
    print("Não existem dados faltantes no DataFrame.")

# Juntando os dataframes
df_final = pd.merge(df_final, dfenso, on=['Mes', 'Ano'])

# Abrindo Novo_CatalogoEstacoesConvencionais CSV 
df_estacoes = pd.read_csv(caminho + 'Novo_CatalogoEstacoesConvencionais.csv')

# Selecionando as colunas 
df_estacoes = df_estacoes[['CD_ESTACAO', 'SG_ESTADO']]

# Converta a coluna 'Estacao' para float e depois para int em df_final
df_final['Estacao'] = df_final['Estacao'].astype(float).astype(int)

# Converta a coluna 'CD_ESTACAO' para float e depois para int em df_estacoes
df_estacoes['CD_ESTACAO'] = df_estacoes['CD_ESTACAO'].astype(float).astype(int)

# Unincdo os DataFrame basedo na Estacao (left) e CD_ESTACAO (right)
df_final = df_final.merge(df_estacoes, left_on='Estacao', right_on='CD_ESTACAO', how='left')

# Criar uma coluna regiao
estados_regiao = {
    'AC': 'Norte',
    'AL': 'Nordeste',
    'AM': 'Norte',
    'AP': 'Norte',
    'BA': 'Nordeste',
    'CE': 'Nordeste',
    'DF': 'Centro-Oeste',
    'ES': 'Sudeste',
    'GO': 'Centro-Oeste',
    'MA': 'Nordeste',
    'MG': 'Sudeste',
    'MS': 'Centro-Oeste',
    'MT': 'Centro-Oeste',
    'PA': 'Norte',
    'PB': 'Nordeste',
    'PE': 'Nordeste',
    'PI': 'Nordeste',
    'PR': 'Sul',
    'RJ': 'Sudeste',
    'RN': 'Nordeste',
    'RO': 'Norte',
    'RR': 'Norte',
    'RS': 'Sul',
    'SC': 'Sul',
    'SE': 'Nordeste',
    'SP': 'Sudeste',
    'TO': 'Norte',
}

df_final['REGIAO'] = df_final['SG_ESTADO'].apply(lambda x: estados_regiao.get(x)).fillna('Indefinida')

# Agrupando os dados por estado, mês e ano
df_agrupado_novo = df_final.groupby(['SG_ESTADO', 'Mes', 'Ano']).agg({'Precipitacao':'sum', 'Temp Comp Media':'mean'}).reset_index()

# Calculando as variações percentuais anuais por estado
df_agrupado_novo['Variação Anual Estado Temp Comp Media'] = df_agrupado_novo.groupby(['SG_ESTADO', 'Mes'])['Temp Comp Media'].pct_change(fill_method=None).round(3)
df_agrupado_novo['Variação Anual Estado Precipitacao'] = df_agrupado_novo.groupby(['SG_ESTADO', 'Mes'])['Precipitacao'].pct_change(fill_method=None).round(3)
df_agrupado_novo = pd.merge(df_agrupado_novo, dfenso, on=['Mes', 'Ano'])

# Agrupando os dados por estado, mês e ano
df_agrupado_novo2 = df_final.groupby(['REGIAO', 'Mes', 'Ano']).agg({'Precipitacao':'sum', 'Temp Comp Media':'mean'}).reset_index()

# Calculando as variações percentuais anuais por regiao
df_agrupado_novo2['Variação Anual Regiao Temp Comp Media'] = df_agrupado_novo2.groupby(['REGIAO', 'Mes'])['Temp Comp Media'].pct_change(fill_method=None).round(3)
df_agrupado_novo2['Variação Anual Regiao Precipitacao'] = df_agrupado_novo2.groupby(['REGIAO', 'Mes'])['Precipitacao'].pct_change(fill_method=None).round(3)
df_agrupado_novo2 = pd.merge(df_agrupado_novo2, dfenso, on=['Mes', 'Ano'])
# Salvar o DataFrame final como um arquivo CSV
df_final.to_csv(caminho + 'dados_ml.csv', index=False)
df = df_final

# Lista de pares de variáveis para calcular a correlação
pares_variaveis_temp = [
    ['Variação Total Anual Temp Comp Media', 'Variação_ONI'],
    ['Temp Comp Media', 'Variação_ONI'],
    ['Variação Total Anual Temp Comp Media', 'ONI'],
    ['Temp Comp Media', 'ONI']
]

pares_variaveis_precip = [
    ['Variação Total Anual Precipitacao', 'Variação_ONI'],
    ['Precipitacao', 'Variação_ONI'],
    ['Variação Total Anual Precipitacao', 'ONI'],
    ['Precipitacao', 'ONI']
]

# Lista para armazenar os resultados
resultados_temp = []
resultados_precip = []

# Loop através dos pares de variáveis para calcular a correlação
for variaveis in pares_variaveis_temp:
    correlacao = df_final[variaveis].corr(method='pearson').iloc[0, 1]
    resultados_temp.append([variaveis, correlacao])

for variaveis in pares_variaveis_precip:
    correlacao = df_final[variaveis].corr(method='pearson').iloc[0, 1]
    resultados_precip.append([variaveis, correlacao])

# Converter a lista de resultados em um DataFrame
df_resultados_temp = pd.DataFrame(resultados_temp, columns=['Pares de Variáveis', 'Correlação'])
df_resultados_precip = pd.DataFrame(resultados_precip, columns=['Pares de Variáveis', 'Correlação'])

# Encontrar o par de variáveis com a maior correlação
melhor_correlacao_temp = df_resultados_temp.loc[df_resultados_temp['Correlação'].idxmax()]
melhor_correlacao_precip = df_resultados_precip.loc[df_resultados_precip['Correlação'].idxmax()]

print(f"O par de variáveis com a maior correlação para temperatura é: {melhor_correlacao_temp['Pares de Variáveis']} com uma correlação de {melhor_correlacao_temp['Correlação']}.")
print(f"O par de variáveis com a maior correlação para precipitação é: {melhor_correlacao_precip['Pares de Variáveis']} com uma correlação de {melhor_correlacao_precip['Correlação']}.")

# Definir a função de correlação
def calculate_correlation(df, variable_name, group_column, variable_column, pair_names):
    # Criar uma copia do df
    df_copy = df.copy()

    # Renomear a coluna baseado na variavel
    base_column_name = f'Variação Total Anual {variable_name} Comp Media'
    new_column_name = variable_column
    df_copy.rename(columns={base_column_name: new_column_name}, inplace=True)

    # Agrupar os dados
    grouped_data = df_copy.groupby(group_column)

    # Calcular a correlacao para cada grupo
    correlations = []
    for group_name, group_df in grouped_data:
        for pair_name in pair_names:
            correlation = group_df[pair_name[0]].corr(group_df[pair_name[1]], method='pearson')
            correlations.append((group_name, pair_name, correlation))

    # Ordenar as correlações por valores absolutos
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    return correlations

# Estacoes (usando df_final)
top_correlations_estacoes_temp = calculate_correlation(df_final, 'temperatura', 'Estacao', 'Temp Comp Media', [
    ['Temp Comp Media', 'Variação_ONI'],
    ['Temp Comp Media', 'ONI']
])

top_correlations_estacoes_precip = calculate_correlation(df_final, 'precipitação', 'Estacao', 'Precipitacao', [
    ['Precipitacao', 'Variação_ONI'],
    ['Precipitacao', 'ONI']
])

# Estados (usando df_dados_agrupados)
top_correlations_estados_temp = calculate_correlation(df_agrupado_novo, 'temperatura', 'SG_ESTADO', 'Temp Comp Media', [
    ['Variação Anual Estado Temp Comp Media', 'Variação_ONI'],
    ['Temp Comp Media', 'Variação_ONI'],
    ['Variação Anual Estado Temp Comp Media', 'ONI'],
    ['Temp Comp Media', 'ONI']
])

top_correlations_estados_precip = calculate_correlation(df_agrupado_novo, 'precipitação', 'SG_ESTADO', 'Precipitacao', [
    ['Variação Anual Estado Precipitacao', 'Variação_ONI'],
    ['Precipitacao', 'Variação_ONI'],
    ['Variação Anual Estado Precipitacao', 'ONI'],
    ['Precipitacao', 'ONI']
])

# Regioes (usando df_dados_agrupados2)
top_correlations_regioes_temp = calculate_correlation(df_agrupado_novo2, 'temperatura', 'REGIAO', 'Temp Comp Media', [
    ['Variação Anual Regiao Temp Comp Media', 'Variação_ONI'],
    ['Temp Comp Media', 'Variação_ONI'],
    ['Variação Anual Regiao Temp Comp Media', 'ONI'],
    ['Temp Comp Media', 'ONI']
])

top_correlations_regioes_precip = calculate_correlation(df_agrupado_novo2, 'precipitação', 'REGIAO', 'Precipitacao', [
    ['Variação Anual Regiao Precipitacao', 'Variação_ONI'],
    ['Precipitacao', 'Variação_ONI'],
    ['Variação Anual Regiao Precipitacao', 'ONI'],
    ['Precipitacao', 'ONI']
])

# Mostrar as melhores correlações por categoria
print("\nTop Correlations for Temperature by Station:")
for correlation in top_correlations_estacoes_temp:
    print(f"{correlation[0]}: {correlation[1]}: {correlation[2]:.4f}")

print("\nTop Correlations for Temperature by State:")
for correlation in top_correlations_estados_temp:
    print(f"{correlation[0]}: {correlation[1]}: {correlation[2]:.4f}")

print("\nTop Correlations for Temperature by Region:")
for correlation in top_correlations_regioes_temp:
    print(f"{correlation[0]}: {correlation[1]}: {correlation[2]:.4f}")

print("\nTop Correlations for Precipitation by Station:")
for correlation in top_correlations_estacoes_precip:
    print(f"{correlation[0]}: {correlation[1]}: {correlation[2]:.4f}")

print("\nTop Correlations for Precipitation by State:")
for correlation in top_correlations_estados_precip:
    print(f"{correlation[0]}: {correlation[1]}: {correlation[2]:.4f}")

print("\nTop Correlations for Precipitation by Region:")
for correlation in top_correlations_regioes_precip:
    print(f"{correlation[0]}: {correlation[1]}: {correlation[2]:.4f}")

# Cria um DataFrame para cada categoria
df_estacoes_temp = pd.DataFrame(top_correlations_estacoes_temp, columns=['Estação', 'Pares de Variáveis', 'Correlação'])
df_estacoes_precip = pd.DataFrame(top_correlations_estacoes_precip, columns=['Estação', 'Pares de Variáveis', 'Correlação'])

df_estados_temp = pd.DataFrame(top_correlations_estados_temp, columns=['Estado', 'Pares de Variáveis', 'Correlação'])
df_estados_precip = pd.DataFrame(top_correlations_estados_precip, columns=['Estado', 'Pares de Variáveis', 'Correlação'])

df_regioes_temp = pd.DataFrame(top_correlations_regioes_temp, columns=['Região', 'Pares de Variáveis', 'Correlação'])
df_regioes_precip = pd.DataFrame(top_correlations_regioes_precip, columns=['Região', 'Pares de Variáveis', 'Correlação'])

# Converta a coluna 'Estação' para string nos DataFrames relevantes
df_estacoes_temp['Estação'] = df_estacoes_temp['Estação'].astype(str)
df_estacoes_precip['Estação'] = df_estacoes_precip['Estação'].astype(str)

# Concatena todos os DataFrames
df_final_correlacao = pd.concat([df_estacoes_temp, df_estacoes_precip, df_estados_temp, df_estados_precip, df_regioes_temp, df_regioes_precip])

# Cria uma coluna com números absolutos
df_estacoes_temp['Correlação Absoluta'] = np.abs(df_estacoes_temp['Correlação'])
df_estacoes_precip['Correlação Absoluta'] = np.abs(df_estacoes_precip['Correlação'])

df_estados_temp['Correlação Absoluta'] = np.abs(df_estados_temp['Correlação'])
df_estados_precip['Correlação Absoluta'] = np.abs(df_estados_precip['Correlação'])

df_regioes_temp['Correlação Absoluta'] = np.abs(df_regioes_temp['Correlação'])
df_regioes_precip['Correlação Absoluta'] = np.abs(df_regioes_precip['Correlação'])

# Retorna os dados com o MAX desse número absoluto por estação, estado, região
df_estacoes_temp_max = df_estacoes_temp.loc[df_estacoes_temp.groupby('Estação')['Correlação Absoluta'].idxmax()]
df_estacoes_precip_max = df_estacoes_precip.loc[df_estacoes_precip.groupby('Estação')['Correlação Absoluta'].idxmax()]

df_estados_temp_max = df_estados_temp.loc[df_estados_temp.groupby('Estado')['Correlação Absoluta'].idxmax()]
df_estados_precip_max = df_estados_precip.loc[df_estados_precip.groupby('Estado')['Correlação Absoluta'].idxmax()]

df_regioes_temp_max = df_regioes_temp.loc[df_regioes_temp.groupby('Região')['Correlação Absoluta'].idxmax()]
df_regioes_precip_max = df_regioes_precip.loc[df_regioes_precip.groupby('Região')['Correlação Absoluta'].idxmax()]

# Cria o mapa de dispersão
plt.figure(figsize=(15, 10))
plt.scatter(df_estacoes_temp_max['Estação'], df_estacoes_temp_max['Correlação'], color='blue')
plt.title('Mapa de Dispersão para Temperatura por Estação')
plt.xlabel('Estação')
plt.ylabel('Correlação')
# Rotaciona os rótulos do eixo x e diminui o tamanho da fonte
plt.xticks(rotation='vertical', fontsize='small')
plt.show()

plt.figure(figsize=(15, 10))
plt.scatter(df_estacoes_precip_max['Estação'], df_estacoes_precip_max['Correlação'], color='blue')
plt.title('Mapa de Dispersão para Precipitação por Estação')
plt.xlabel('Estação')
plt.ylabel('Correlação')
# Rotaciona os rótulos do eixo x e diminui o tamanho da fonte
plt.xticks(rotation='vertical', fontsize='small')
plt.show()

# Cria um mapa de dispersão para temperatura para cada Estado
plt.figure(figsize=(10, 6))
plt.scatter(df_estados_temp_max['Estado'], df_estados_temp_max['Correlação'], color='red')
plt.title('Mapa de Dispersão para Temperatura por Estado')
plt.xlabel('Estado')
plt.ylabel('Correlação')
plt.show()

# Cria um mapa de dispersão para precipitação para cada Estado
plt.figure(figsize=(10, 6))
plt.scatter(df_estados_precip_max['Estado'], df_estados_precip_max['Correlação'], color='red')
plt.title('Mapa de Dispersão para Precipitação por Estado')
plt.xlabel('Estado')
plt.ylabel('Correlação')
plt.show()

# Cria um mapa de dispersão para temperatura para cada Região
plt.figure(figsize=(10, 6))
plt.scatter(df_regioes_temp_max['Região'], df_regioes_temp_max['Correlação'], color='green')
plt.title('Mapa de Dispersão para Temperatura por Região')
plt.xlabel('Região')
plt.ylabel('Correlação')
plt.show()

# Cria um mapa de dispersão para precipitação para cada Região
plt.figure(figsize=(10, 6))
plt.scatter(df_regioes_precip_max['Região'], df_regioes_precip_max['Correlação'], color='green')
plt.title('Mapa de Dispersão para Precipitação por Região')
plt.xlabel('Região')
plt.ylabel('Correlação')
plt.show()

# Exporta o DataFrame final para um arquivo CSV
df_final_correlacao.to_csv(caminho + 'correlacoes.csv', index=False)

df_final.to_csv(caminho + 'correlacao_dados_estacao.csv', index=False)
df_agrupado_novo.to_csv(caminho + 'correlacao_dados_estado.csv', index=False)
df_agrupado_novo2.to_csv(caminho + 'correlacao_dados_regiao.csv', index=False)

# Obtém os valores absolutos da correlação
df_final_correlacao['Correlação Absoluta'] = df_final_correlacao['Correlação'].apply(np.abs)

# Ordena o DataFrame pela correlação absoluta em ordem decrescente
df_final_correlacao = df_final_correlacao.sort_values(by='Correlação Absoluta', ascending=False)

# Obtém as 5 primeiras estações
primeiras_estacoes = df_final_correlacao['Estação'].dropna().unique()[:2]

# Obtém os 3 primeiros estados
primeiros_estados = df_final_correlacao['Estado'].dropna().unique()[:2]

# Obtém a primeira região
primeira_regiao = df_final_correlacao['Região'].dropna().unique()[:1]

print(primeiras_estacoes)
print(primeiros_estados)
print(primeira_regiao)

# Convertendo lista de estações para INT
primeiras_estacoes_int = [int(estacao) for estacao in primeiras_estacoes]

#Filtrando 
regressao_estacoes = df_final.loc[df_final['Estacao'].dropna().isin(primeiras_estacoes_int).reset_index(drop=True)]
regressao_estados = df_agrupado_novo.loc[df_agrupado_novo['SG_ESTADO'].dropna().isin(primeiros_estados).reset_index(drop=True)]
regressao_regiao = df_agrupado_novo2.loc[df_agrupado_novo2['REGIAO'].dropna().isin(primeira_regiao).reset_index(drop=True)]

stations = regressao_estacoes['Estacao'].unique()  # Retorna as estações
estados = regressao_estados['SG_ESTADO'].unique()  # Retorna os estados
regiao = regressao_regiao['REGIAO'].unique()  # Retorna a regiao

for station in stations:
    # Filtrar por estação
    station_data = regressao_estacoes[regressao_estacoes['Estacao'] == station]

    # Substituir Inf por NaN
    station_data = station_data.replace([np.inf, -np.inf], np.nan)

    # Excluir linhas com NaN
    station_data = station_data.dropna(subset=['Temp Comp Media', 'Variação_ONI'])

    # Preparando os dados
    y = station_data['Temp Comp Media'].values
    X = station_data[['Ano','Variação_ONI']].values

    # Criando o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicao
    predicted_y = model.predict(X)

    # Calculando o MSE e o R²
    mse = mean_squared_error(y, predicted_y)
    r2 = r2_score(y, predicted_y)

    print("Estações: Temperatura")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")

    # Criando grafico
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, label='Observed', alpha=0.7)  # Ano (year) on x-axis
    plt.plot(X[:, 0], predicted_y, label='Predicted', color='red')
    plt.xlabel('Ano')
    plt.ylabel('Temp Comp Media')
    plt.title(f"Temperatura - Station: {station}")
    plt.legend()
    plt.grid(True)
    plt.show()


for station in stations:
    # Filtrar por estação
    station_data = regressao_estacoes[regressao_estacoes['Estacao'] == station]

    # Substituir Inf por NaN
    station_data = station_data.replace([np.inf, -np.inf], np.nan)

    # Excluir linhas com NaN
    station_data = station_data.dropna(subset=['Precipitacao', 'ONI'])

    # Preparando os dados
    y = station_data['Precipitacao'].values
    X = station_data[['Ano','ONI']].values

    # Criando o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicao
    predicted_y = model.predict(X)

    # Calculando o MSE e o R²
    mse = mean_squared_error(y, predicted_y)
    r2 = r2_score(y, predicted_y)

    print("Estações: Precipitação")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")
    
    # Criando grafico
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, label='Observed', alpha=0.7)  # Ano (year) on x-axis
    plt.plot(X[:, 0], predicted_y, label='Predicted', color='red')
    plt.xlabel('Ano')
    plt.ylabel('Precipitacao')
    plt.title(f"Precipitação - Station: {station}")
    plt.legend()
    plt.grid(True)
    plt.show()

for estado in estados:
    # Filtrar por estação
    estado_data = regressao_estados[regressao_estados['SG_ESTADO'] == estado]

    # Substituir Inf por NaN
    estado_data = estado_data.replace([np.inf, -np.inf], np.nan)

    # Excluir linhas com NaN
    estado_data = estado_data.dropna(subset=['Variação Anual Estado Temp Comp Media', 'Variação_ONI'])

    # Preparando os dados
    y = estado_data['Variação Anual Estado Temp Comp Media'].values
    X = estado_data[['Ano','Variação_ONI']].values

    # Criando o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicao
    predicted_y = model.predict(X)

    # Calculando o MSE e o R²
    mse = mean_squared_error(y, predicted_y)
    r2 = r2_score(y, predicted_y)

    print("Estados: Temperatura")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")

    # Criando grafico
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, label='Observed', alpha=0.7)  # Ano (year) on x-axis
    plt.plot(X[:, 0], predicted_y, label='Predicted', color='red')
    plt.xlabel('Ano')
    plt.ylabel('Variação Anual Estado Temp Comp Media')
    plt.title(f"Temperatura - Estado: {estado}")
    plt.legend()
    plt.grid(True)
    plt.show()

for estado in estados:
    # Filtrar por estação
    estado_data = regressao_estados[regressao_estados['SG_ESTADO'] == estado]

    # Substituir Inf por NaN
    estado_data = estado_data.replace([np.inf, -np.inf], np.nan)

    # Excluir linhas com NaN
    estado_data = estado_data.dropna(subset=['Precipitacao', 'ONI'])

    # Preparando os dados
    y = estado_data['Precipitacao'].values
    X = estado_data[['Ano','ONI']].values

    # Criando o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicao
    predicted_y = model.predict(X)

    # Calculando o MSE e o R²
    mse = mean_squared_error(y, predicted_y)
    r2 = r2_score(y, predicted_y)

    print("Estados: Temperatura")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")

    # Criando grafico
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, label='Observed', alpha=0.7)  # Ano (year) on x-axis
    plt.plot(X[:, 0], predicted_y, label='Predicted', color='red')
    plt.xlabel('Ano')
    plt.ylabel('Precipitacao')
    plt.title(f"Precipitação - Estado: {estado}")
    plt.legend()
    plt.grid(True)
    plt.show()

lista_regioes = regiao
for regiao in lista_regioes:
    # Filtrar por estação
    regiao_data = regressao_regiao[regressao_regiao['REGIAO'] == regiao]
    
    # Substituir Inf por NaN
    regiao_data = regiao_data.replace([np.inf, -np.inf], np.nan)

    # Excluir linhas com NaN
    regiao_data = regiao_data.dropna(subset=['Temp Comp Media', 'ONI'])

    # Preparando os dados
    y = regiao_data['Temp Comp Media'].values
    X = regiao_data[['Ano','ONI']].values

    # Criando o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicao
    predicted_y = model.predict(X)

    # Calculando o MSE e o R²
    mse = mean_squared_error(y, predicted_y)
    r2 = r2_score(y, predicted_y)

    print("Região: Temperatura")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")

    # Criando grafico
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, label='Observed', alpha=0.7)  # Ano (year) on x-axis
    plt.plot(X[:, 0], predicted_y, label='Predicted', color='red')
    plt.xlabel('Ano')
    plt.ylabel('Temp Comp Media')
    plt.title(f"Temperatura - Região: {regiao}")
    plt.legend()
    plt.grid(True)
    plt.show()

for regiao in lista_regioes:
    # Filtrar por estação
    # regiao_data = regressao_regiao[regressao_regiao['REGIAO'] == regiao]

    # Substituir Inf por NaN
    # regiao_data = regiao_data.replace([np.inf, -np.inf], np.nan)

    # Excluir linhas com NaN
    regiao_data = regiao_data.dropna(subset=['Variação Anual Regiao Precipitacao', 'Variação_ONI'])
    regiao_data.to_csv(caminho + 'teste.csv')
    # Preparando os dados
    y = regiao_data['Variação Anual Regiao Precipitacao'].values
    X = regiao_data[['Ano','Variação_ONI']].values

    # Criando o modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicao
    predicted_y = model.predict(X)

    # Calculando o MSE e o R²
    mse = mean_squared_error(y, predicted_y)
    r2 = r2_score(y, predicted_y)

    print("Região: Precipitação")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")

    # Criando grafico
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, label='Observed', alpha=0.7)  # Ano (year) on x-axis
    plt.plot(X[:, 0], predicted_y, label='Predicted', color='red')
    plt.xlabel('Ano')
    plt.ylabel('Variação Anual Regiao Precipitacao')
    plt.title(f"Precipitação - Região: {regiao}")
    plt.legend()
    plt.grid(True)
    plt.show()
