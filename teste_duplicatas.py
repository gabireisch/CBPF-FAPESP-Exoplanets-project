import pandas as pd

def verificar_duplicatas(csv_path):
    print(f"Analisando arquivo: {csv_path}")
    
    try:
        # Tenta ler com vírgula, se falhar, tenta com ponto e vírgula
        df = pd.read_csv(csv_path)
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        print(f"Erro ao abrir o CSV {csv_path}: {e}")
        return

    # Pega o nome da primeira coluna (para evitar aquele KeyError)
    nome_primeira_coluna = df.columns[0]
    
    # Extrai os IDs e remove valores em branco (NaNs)
    ids = df[nome_primeira_coluna].dropna().astype(int)
    
    total_linhas = len(ids)
    ids_unicos = len(set(ids))
    duplicatas = total_linhas - ids_unicos
    
    print(f"Total de linhas lidas: {total_linhas}")
    print(f"Total de TIC IDs únicos: {ids_unicos}")
    
    if duplicatas == 0:
        print("\nOk, sem duplicados.")
    else:
        print(f"\nALERTA: Foram encontradas {duplicatas} duplicatas!")
        
        # O value_counts() conta quantas vezes cada número aparece
        contagem = ids.value_counts()
        repetidos = contagem[contagem > 1]
        
        print("\nAqui estão os TICs 'vazadores' (e quantas vezes aparecem no CSV):")
        print(repetidos)

# Rode para os dois arquivos do seu projeto
verificar_duplicatas("dados/positivos.csv")
verificar_duplicatas("dados/tess_ebcatalog.csv")
verificar_duplicatas("dados/negativos.csv")