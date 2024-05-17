import pandas as pd
import random

# Caminho dos arquivos
caminho = 'C:\\TCC\\bases_tempo\\'

# Lista de arquivos CSV
arquivos_base = ['conventional_weather_stations_inmet_brazil_1961_2019.csv', 'CatalogoEstaçõesConvencionais.csv', 'dados_2020_2024.csv', 'ENSO.csv']
arquivos_finais = ['dados_finais.csv', 'dados_ml.csv', 'Novo_CatalogoEstacoesConvencionais.csv']

# Dicionário com os nomes das colunas esperadas para cada arquivo
colunas_esperadas = {
    'conventional_weather_stations_inmet_brazil_1961_2019.csv': ['Estacao', 'Data', 'Hora', 'Precipitacao', 'TempBulboSeco', 'TempBulboUmido', 'TempMaxima', 'TempMinima', 'UmidadeRelativa', 'PressaoAtmEstacao', 'PressaoAtmMar', 'DirecaoVento', 'VelocidadeVento', 'Insolacao', 'Nebulosidade', 'Evaporacao Piche', 'Temp Comp Media', 'Umidade Relativa Media', 'Velocidade do Vento Media'],
    'CatalogoEstaçõesConvencionais.csv': ['DC_NOME', 'SG_ESTADO', 'CD_SITUACAO', 'VL_LATITUDE', 'VL_LONGITUDE', 'VL_ALTITUDE', 'DT_INICIO_OPERACAO', 'CD_ESTACAO'],
    'dados_2020_2024.csv': ['Periodicidade da Medicao', 'Data Final', 'Data Inicial', 'Situacao', 'Altitude', 'Longitude', 'Latitude', 'estacao', 'Nome', 'Data Medicao', 'EVAPORACAO DO PICHE, DIARIA(mm)', 'INSOLACAO TOTAL, DIARIO(h)', 'PRECIPITACAO TOTAL, DIARIO(mm)', 'TEMPERATURA MAXIMA, DIARIA(°C)', 'TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)', 'TEMPERATURA MINIMA, DIARIA(°C)', 'UMIDADE RELATIVA DO AR, MEDIA DIARIA(%)', 'UMIDADE RELATIVA DO AR, MINIMA DIARIA(%)', 'VENTO, VELOCIDADE MEDIA DIARIA(m/s)'],
    'ENSO.csv': ['Date', 'Year', 'Month', 'Global Temperature Anomalies', 'Nino 1+2 SST', 'Nino 1+2 SST Anomalies', 'Nino 3 SST', 'Nino 3 SST Anomalies', 'Nino 3.4 SST', 'Nino 3.4 SST Anomalies', 'Nino 4 SST', 'Nino 4 SST Anomalies', 'TNI', 'PNA', 'OLR', 'SOI', 'Season (2-Month)', 'MEI.v2', 'Season (3-Month)', 'ONI', 'Season (12-Month)', 'ENSO Phase-Intensity'],
    'dados_finais.csv': ['Estacao', 'Data', 'Evaporacao Piche', 'Insolacao', 'Precipitacao', 'TempMaxima', 'Temp Comp Media', 'TempMinima', 'Umidade Relativa Media', 'Velocidade do Vento Media'],
    'dados_ml.csv': ['Estacao', 'Mes', 'Ano', 'Precipitacao', 'Temp Comp Media', 'Variação Anual Estação Temp Comp Media', 'Variação Anual Estação Precipitacao', 'Variação Total Anual Temp Comp Media', 'Variação Total Anual Precipitacao', 'ONI', 'ONI_ajustado', 'Variação_ONI', 'CD_ESTACAO', 'SG_ESTADO', 'REGIAO'],
    'Novo_CatalogoEstacoesConvencionais.csv': ['DC_NOME', 'SG_ESTADO', 'CD_SITUACAO', 'VL_LATITUDE', 'VL_LONGITUDE', 'VL_ALTITUDE', 'DT_INICIO_OPERACAO', 'CD_ESTACAO']
}


# Verificar cada arquivo final
for arquivo_final in arquivos_finais:
    df_final = pd.read_csv(caminho + arquivo_final, delimiter=',')
    colunas_arquivo_final = df_final.columns.tolist()
    
    # Verificar se os dados estão de acordo
    for coluna in colunas_arquivo_final:
        # Selecionar uma amostra aleatória de dados para verificar
        amostra = df_final[coluna].sample(n=10)
        
        # Verificar se os dados da amostra estão presentes nos arquivos base
        for arquivo_base in arquivos_base:
            # O arquivo ENSO.csv é separado por vírgulas
            if arquivo_base == 'ENSO.csv':
                df_base = pd.read_csv(caminho + arquivo_base, delimiter=',')
            else:
                df_base = pd.read_csv(caminho + arquivo_base, delimiter=';')
            
            if coluna in df_base.columns:
                esta_presente = df_base[coluna].isin(amostra).any()
                print(f'Os dados da coluna {coluna} do arquivo final {arquivo_final} estão presentes no arquivo base {arquivo_base}: {esta_presente}')