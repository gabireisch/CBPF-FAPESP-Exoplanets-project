import os
import random
import shutil
from pathlib import Path
import pandas as pd
from lightkurve import search_lightcurve


#configuracao de divisao dos dados
RANDOM_SEED = 42
TRAIN_PERCENT = 0.70
VALIDATION_PERCENT = 0.15
TEST_PERCENT = 0.15
BASE_DIR = "dados"
random.seed(RANDOM_SEED)

def split_tic_ids(tic_ids):
    random.shuffle(tic_ids)
    n_total = len(tic_ids)
    n_train = int(n_total * TRAIN_PERCENT)
    n_validation = int(n_total * VALIDATION_PERCENT)
    train_ids = tic_ids[:n_train]
    validation_ids = tic_ids[n_train:n_train + n_validation]
    test_ids = tic_ids[n_train + n_validation:]

    return train_ids, validation_ids, test_ids


#download de dados
def baixar_classe(csv_path, class_name):
    print(f"Classe: {class_name}")

    # lê csv
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["TIC ID"])
    tic_ids = df["TIC ID"].astype(int).tolist()

    print(f"{len(tic_ids)} TIC IDs encontrados")

    # split
    train_ids, validation_ids, test_ids = split_tic_ids(tic_ids)
    splits = {"train": train_ids,"validation": validation_ids,"test": test_ids}

    # cria diretórios
    for split_name in splits.keys():
        path = os.path.join(BASE_DIR,split_name,class_name)
        os.makedirs(path, exist_ok=True)

    # download
    for split_name, ids in splits.items():
        print(f"\n{split_name.upper()} -> {len(ids)} arquivos")
        for tic in ids:
            tic_name = f"TIC {tic}"
            print(f"\nProcessando {tic_name}")

            try:
                # busca SPOC
                search_result = search_lightcurve(tic_name,mission="TESS",author="SPOC")
                if len(search_result) == 0:
                    print("Nenhuma light curve encontrada")
                    continue

                # baixa só o primeiro setor
                lc = search_result[0].download()
                if lc is None:
                    print("Falha no download")
                    continue

                # remove NaNs
                lc = lc.remove_nans()

                # nome
                filename = f"TIC_{tic}.fits"
                save_path = os.path.join(BASE_DIR,split_name,class_name,filename)
                lc.to_fits(save_path)

            except Exception as e: 
                print(f"Erro em {tic_name}")
                print(e)

#execucao
baixar_classe("dados/positivos.csv", "positives")
baixar_classe("dados/negativos.csv", "negatives")
print("\nDownload concluído.")