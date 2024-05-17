import pandas as pd
import os
import csv

def replace_comma(x):
    if isinstance(x, (int, float, str, object)):
        return str(x).replace(',', '.')
    elif isinstance(x, int, float, str, object) and x.lower() in ['nan', 'null', '']:
        return '0'  # Substitui dados faltantes por '0'
    else:
        return x

import pandas as pd
import os

# Função para ler um arquivo CSV
def read_csv_file(file_path):
    # Abrindo o arquivo no modo de leitura com codificação 'utf-8'
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Extraindo os metadados das primeiras 9 linhas do arquivo
    metadata = {line.split(':')[0]: line.split(':')[1].strip() for line in lines[:9]}  
    
    # Lendo o arquivo CSV em um DataFrame do pandas
    data = pd.read_csv(file_path, delimiter=';', skiprows=11, header=None, encoding='utf-8')  
    
    # Verificando se o número de colunas no DataFrame corresponde ao número de elementos na linha 10 do arquivo
    if len(data.columns) == len(lines[10].split(';')):
        # Definindo os nomes das colunas para os valores na linha 10
        data.columns = lines[10].split(';')
    
    # Inserindo os metadados no DataFrame
    for key, value in metadata.items():
        data.insert(0, key, value)  
    
    # Renomeando a coluna 'Codigo Estacao' para 'estacao'
    data.rename(columns={'Codigo Estacao': 'estacao'}, inplace=True)  
    
    # Aplicando a função replace_comma para substituir ',' por '.' em todas as colunas
    data = data.apply(lambda col: col.map(replace_comma))

    # Retornando o DataFrame
    return data

# Função para processar o DataFrame após a combinação
def process_all_data(all_data):
    """
    Aplica a função replace_comma para substituir ',' por '.' e 'nan', 'null' e '' por '0' em todas as colunas do DataFrame.

    Args:
        all_data (pandas.DataFrame): O DataFrame contendo todos os dados combinados dos arquivos CSV.

    Returns:
        pandas.DataFrame: O DataFrame processado com as substituições feitas.
    """
    return all_data.apply(lambda col: col.map(replace_comma))

# Função para combinar vários arquivos CSV em um único arquivo
def combine_csv_files(folder_path, output_file):
    # Listando todos os arquivos no diretório especificado que terminam com '.csv'
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    
    # Verificando se nenhum arquivo CSV foi encontrado
    if not files:
        print(f'Nenhum arquivo CSV encontrado em {folder_path}')
        return
    
    # Para cada arquivo CSV
    for i, f in enumerate(files):
        try:
            # Lendo o arquivo usando a função read_csv_file e anexando-o ao arquivo de saída
            df = read_csv_file(os.path.join(folder_path, f))
            if i == 0:
                df.to_csv(output_file, sep=';', index=False)
            else:
                df.to_csv(output_file, sep=';', mode='a', header=False, index=False)
        except Exception as e:
            # Imprimindo uma mensagem de erro se ocorrer um erro ao ler um arquivo
            print(f'Erro ao ler o arquivo {f}: {e}')

# Definindo o caminho da pasta contendo os arquivos CSV
folder_path = 'C:\\\\TCC\\\\bases_tempo\\\\base_tempo_2020-2023'

# Definindo o caminho do arquivo de saída
caminho = 'C:\\TCC\\bases_tempo\\'
output_file = caminho + 'dados_2020_2024.csv'

# Chamando a função combine_csv_files para combinar todos os arquivos CSV em um único arquivo
combine_csv_files(folder_path, output_file)

# Caminho do arquivo
basehistorica = caminho + 'conventional_weather_stations_inmet_brazil_1961_2019.csv'
baserecente = caminho + 'dados_2020_2024.csv'

# Ler os arquivos CSV
dfhist = pd.read_csv(basehistorica, sep = ';')
dfnovo = pd.read_csv(baserecente, encoding='utf-8', quoting=csv.QUOTE_NONE, sep = ';')

# Encontrar valores únicos na coluna 'Estacao' nos dois dataframes
valores_unicos1 = dfhist['Estacao'].unique()
valores_unicos3 = dfnovo['estacao'].unique()

# Obter a interseção dos valores únicos
valores_unicos = list(set(valores_unicos1) & set(valores_unicos3))

# Imprimir os valores únicos
for valor in valores_unicos:
    print(valor)

# Caminho do arquivo
catalogoestacoes = caminho + 'CatalogoEstaçõesConvencionais.csv'

# Ler o segundo arquivo CSV
dfcatalogo = pd.read_csv(catalogoestacoes, sep=';')

# Filtrar as linhas onde 'CD_ESTACAO' está presente em 'valores_unicos'
dfcatalogo_filtrado = dfcatalogo[dfcatalogo['CD_ESTACAO'].isin(valores_unicos)]

# Salvar o novo arquivo CSV
dfcatalogo_filtrado.to_csv(caminho + 'Novo_CatalogoEstacoesConvencionais.csv', index=False)

# Converter a coluna 'Data' para o tipo datetime
dfnovo['Data Medicao'] = pd.to_datetime(dfnovo['Data Medicao'])

# Alterar o formato da data para 'dia/mês/ano'
dfnovo['Data Medicao'] = dfnovo['Data Medicao'].dt.strftime('%d/%m/%Y')

# Renomeie a coluna para fazer a junção
dfcatalogo_filtrado = dfcatalogo_filtrado.rename(columns={'CD_ESTACAO': 'Estacao'})

# Filtre dfhist com base em dfnovabase
dfhist = dfhist[dfhist['Estacao'].isin(dfcatalogo_filtrado['Estacao'])]

# Agrupe os dados e calcule as estatísticas necessárias
dfhist_filtrado = dfhist.groupby(['Estacao', 'Data']).agg({
    'Precipitacao': 'sum',
    'TempMaxima': 'max',
    'TempMinima': 'min',
    'Temp Comp Media': 'mean',
    'Precipitacao': 'sum',
    'TempBulboSeco': 'mean',
    'TempBulboUmido': 'mean',
    'UmidadeRelativa': 'mean',
    'PressaoAtmEstacao': 'mean',
    'PressaoAtmMar': 'mean',
    'DirecaoVento': 'first',
    'VelocidadeVento': 'mean',
    'Insolacao': 'mean',
    'Nebulosidade': 'mean',
    'Evaporacao Piche': 'mean',
    'Umidade Relativa Media': 'mean',
    'Velocidade do Vento Media': 'mean'
}).reset_index()

# Filtre dfnovo com base em dfnovabase
dfnovo_filtrado = dfnovo[dfnovo['estacao'].isin(dfcatalogo_filtrado['Estacao'])]

# Renomeie as colunas no primeiro dataframe
dfnovo_filtrado = dfnovo_filtrado.rename(columns={
    'estacao': 'Estacao',
    'Data Medicao': 'Data',
    'PRECIPITACAO TOTAL, DIARIO(mm)': 'Precipitacao',
    'TEMPERATURA MAXIMA, DIARIA(°C)': 'TempMaxima',
    'TEMPERATURA MINIMA, DIARIA(°C)': 'TempMinima',
    'TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)': 'Temp Comp Media',
    'EVAPORACAO DO PICHE, DIARIA(mm)' : 'Evaporacao Piche',
    'INSOLACAO TOTAL, DIARIO(h)' : 'Insolacao',
    'UMIDADE RELATIVA DO AR, MEDIA DIARIA(%)' : 'Umidade Relativa Media',
    'VENTO, VELOCIDADE MEDIA DIARIA(m/s)' : 'Velocidade do Vento Media',
})

# Encontre as colunas comuns entre os dois dataframes
common_columns = dfnovo_filtrado.columns.intersection(dfhist_filtrado.columns)

# Faça a junção dos dois conjuntos de dados apenas nas colunas comuns
df = pd.merge(dfnovo_filtrado[common_columns], dfhist_filtrado[common_columns], how='outer')
df = df.fillna(0)

# Salve o resultado em um novo arquivo CSV
df.to_csv(caminho + 'dados_finais.csv', index=False)