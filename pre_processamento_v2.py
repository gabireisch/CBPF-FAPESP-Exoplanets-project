import os
import random
import pandas as pd
from lightkurve import search_lightcurve

RANDOM_SEED = 42
TRAIN_PERCENT = 0.70
VALIDATION_PERCENT = 0.15
BASE_DIR = "dados_multisetor" # Nova pasta para não misturar
random.seed(RANDOM_SEED)

def split_tic_ids(tic_ids):
    random.shuffle(tic_ids)
    n_total = len(tic_ids)
    n_train = int(n_total * TRAIN_PERCENT)
    n_validation = int(n_total * VALIDATION_PERCENT)
    return tic_ids[:n_train], tic_ids[n_train:n_train + n_validation], tic_ids[n_train + n_validation:]

def baixar_classe_multi_setor(csv_path, class_name):
    print(f"--- Iniciando download: {class_name} ---")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["TIC ID"])
    tic_ids = df["TIC ID"].astype(int).tolist()

    train_ids, validation_ids, test_ids = split_tic_ids(tic_ids)
    splits = {"train": train_ids, "validation": validation_ids, "test": test_ids}

    for split_name in splits.keys():
        os.makedirs(os.path.join(BASE_DIR, split_name, class_name), exist_ok=True)

    for split_name, ids in splits.items():
        print(f"\nBaixando {split_name.upper()}...")
        for tic in ids:
            tic_name = f"TIC {tic}"
            try:
                # Busca todos os setores em cadência curta (2 min)
                search_result = search_lightcurve(tic_name, mission="TESS", author="SPOC", exptime="short")
                
                # Se não achar, pula
                if len(search_result) == 0:
                    continue

                # O SEGREDO DO ORIENTADOR: Loop por TODOS os setores disponíveis!
                for idx, item in enumerate(search_result):
                    try:
                        lc = item.download()
                        if lc is None: continue
                        
                        lc = lc.remove_nans()
                        sector = lc.meta.get('SECTOR', f'X{idx}') # Tenta pegar o número do setor
                        
                        # Salva como arquivos separados: ex -> TIC_123_sec4.fits
                        filename = f"TIC_{tic}_sec{sector}.fits"
                        save_path = os.path.join(BASE_DIR, split_name, class_name, filename)
                        lc.to_fits(save_path, overwrite=True)
                        
                    except Exception as e_interno:
                        pass # Ignora erros de um setor específico e tenta o próximo
                        
            except Exception as e: 
                print(f"Erro ao buscar {tic_name}")

baixar_classe_multi_setor("dados/positivos.csv", "positives")
baixar_classe_multi_setor("dados/negativos.csv", "negatives")
print("\nDownload Multi-Setor Concluído!")