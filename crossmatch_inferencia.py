import pandas as pd

# 1. Carregar os IDs conhecidos
df_pos = pd.read_csv("dados/positivos.csv")
df_neg = pd.read_csv("dados/negativos.csv")
df_eb = pd.read_csv("dados/tess_ebcatalog.csv", sep=";")  

# Removendo espaços em branco dos nomes das colunas
df_pos.columns = df_pos.columns.str.strip()
df_neg.columns = df_neg.columns.str.strip()
df_eb.columns = df_eb.columns.str.strip()

# Extraindo os IDs das listas originais
tics_pos = set(df_pos["TIC ID"].dropna().astype(int))
tics_neg = set(df_neg["TIC ID"].dropna().astype(int))

# Identificar o nome da coluna de ID no catálogo de EB 
if "TIC ID" in df_eb.columns:
    col_eb = "TIC ID"
elif "TIC" in df_eb.columns:
    col_eb = "TIC"
else:
    col_eb = df_eb.columns[0] # Caso tenha outro nome, assume a primeira coluna

# Limpeza extra: converte para string, remove qualquer ';' que tenha sobrado na coluna, depois converte para int
tics_eb_limpos = df_eb[col_eb].dropna().astype(str).str.replace(";", "", regex=False)
tics_eb = set(tics_eb_limpos.astype(int))

# Unindo os TICs conhecidos (Positivos + Negativos + Binárias Eclipsantes)
tics_conhecidos = tics_pos.union(tics_neg).union(tics_eb)

# 2. Carregar o novo CSV do Setor 84
novo_csv_path = "dados/s0084.csv" 
df_s84 = pd.read_csv(novo_csv_path)

# Limpar colunas e extrair a coluna do Setor 84
df_s84.columns = df_s84.columns.str.strip()
tics_s84 = set(df_s84["#TIC_ID"].dropna().astype(int))

# 3. Encontrar os TICs novos
tics_para_inferencia = tics_s84 - tics_conhecidos

# 4. Resultados no terminal
print(f"Total de TICs extraídos do Setor 84: {len(tics_s84)}")
print(f"Total de TICs conhecidos (Pos + Neg + EB): {len(tics_conhecidos)}")
print(f"TICs novos restantes para inferência: {len(tics_para_inferencia)}")

# 5. Salvar o DataFrame filtrado com os dados de RA e DEC originais
df_filtrado = df_s84[df_s84["#TIC_ID"].astype(int).isin(tics_para_inferencia)]

# Exportando o arquivo final pronto para a inferência
df_filtrado.to_csv("s0084_para_inferencia.csv", index=False)