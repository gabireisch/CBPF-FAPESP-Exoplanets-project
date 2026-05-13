from lightkurve import (search_lightcurve,LightCurveCollection)
import os
import pandas as pd

# cria pasta
os.makedirs("dados/train/negatives", exist_ok=True)

# lê csv
df = pd.read_csv("dados/negativos.csv")

# remove espaços extras dos nomes das colunas
df.columns = df.columns.str.strip()

# mostra colunas para debug
print(df.columns.tolist())

# remove TIC IDs vazios
df = df.dropna(subset=["TIC ID"])

# converte para inteiro
tic_ids = df["TIC ID"].astype(int).tolist()
print(f"{len(tic_ids)} TIC IDs encontrados")

for tic in tic_ids:
    tic_name = f"TIC {tic}"
    print(f"\nProcessando {tic_name}")

    try:
        # busca light curves TESS
        search_result = search_lightcurve(tic_name,mission="TESS", author="SPOC")

        # verifica se encontrou algo
        if len(search_result) == 0:
            print("Nenhuma light curve encontrada")
            continue

        # baixa todos os setores
        lcc = search_result.download_all()
        if lcc is None:
            print("Falha no download")
            continue

        # remove light curves incompatíveis
        valid_lcs = []

        for lc_item in lcc:
            try:
                # força acesso ao quality
                _ = lc_item.quality
                valid_lcs.append(lc_item)

            except Exception:
                print("Light curve incompatível ignorada")

        # junta setores
        lc = LightCurveCollection(valid_lcs).stitch()
        # caminho do arquivo
        filename = f"TIC_{tic}.fits"
        path = os.path.join("dados/train/negatives",filename)

        # salva
        lc.to_fits(path)
        print(f"Salvo em: {path}")

    except Exception as e:
        print(f"Erro em {tic_name}")
        print(e)
