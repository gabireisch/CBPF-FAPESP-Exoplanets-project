import os
import pandas as pd
from lightkurve import search_lightcurve

BASE_DIR = "dados_inferencia_teste"
SETOR_ALVO = 84 # setor escolhido

def baixar_dados_inferencia(csv_path):
    print(f"\nIniciando download a partir de: {csv_path}")
    
    #Leitura do CSV
    try:
        df = pd.read_csv(csv_path)
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        print(f"Erro ao abrir o CSV {csv_path}: {e}")
        return

    #Pega os TIC IDs únicos
    nome_primeira_coluna = df.columns[0]
    df = df.dropna(subset=[nome_primeira_coluna])
    tic_ids = list(set(df[nome_primeira_coluna].astype(int).tolist()))
    print(f"{len(tic_ids)} TIC IDs únicos encontrados para inferência.")

    # Cria o diretório de destino
    os.makedirs(BASE_DIR, exist_ok=True)

    print("\nBaixando curvas de luz...")
    for tic in tic_ids:
        tic_name = f"TIC {tic}"
        try:
            search_result = search_lightcurve(tic_name, mission="TESS", author="SPOC", exptime="short", sector=SETOR_ALVO)
            
            # Se a estrela não foi observada no Setor 84, pula.
            if len(search_result) == 0:
                print(f"Sem dados no Setor {SETOR_ALVO} para {tic_name}")
                continue

            # Como já filtramos, pegamos direto o primeiro resultado
            try:
                lc = search_result[0].download()
                if lc is not None: 
                    # Salva direto como FITS
                    filename = f"TIC_{tic}_sec{SETOR_ALVO}.fits"
                    save_path = os.path.join(BASE_DIR, filename)
                    
                    lc.to_fits(save_path, overwrite=True)
                    print(f"Sucesso: {filename} salvo.")
                else:
                    print(f"Falha ao baixar os dados de {tic_name}")
                    
            except Exception as e_interno:
                 print(f"Erro ao baixar/salvar {tic_name}: {e_interno}")
                    
        except Exception as e: 
            print(f"Erro geral na busca de {tic_name}: {e}")

# Execução 
baixar_dados_inferencia("dados_inferencia_teste/lista_teste_de_inferencia.csv")
print("\nDownload do Setor 84 concluído!")