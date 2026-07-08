import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from qusi.data import FiniteStandardLightCurveDataset, LightCurveCollection
from qusi.model import Hadryss
from qusi.session import get_device, infer_session
caminho_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(caminho_raiz)
from dataset import load_times_and_fluxes_from_path

def get_infer_paths():
    csv_path = os.path.join(caminho_raiz, "CBPF-FAPESP-Exoplanets-project", "s0084_para_inferencia.csv")
    print(f"Lendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    nome_primeira_coluna = df.columns[0]
    
    tic_ids = df[nome_primeira_coluna].dropna().astype(int).unique()
    print(f"Total de TICs no CSV: {len(tic_ids)}")
    
    pasta_dados = Path("/tf/astrodados3/marianab/s0084/")
    print("Indexando arquivos FITS da pasta...")
    
    todos_arquivos = {arquivo.name: arquivo for arquivo in pasta_dados.glob("*.fits")}
    print(f"Total de FITS encontrados na pasta: {len(todos_arquivos)}")
    
    arquivos = []
    
    for i, tic in enumerate(tic_ids):
        if i % 1000 == 0:
            print(f"Processando {i}/{len(tic_ids)} TICs")
        
        tic_str = f"{tic:016d}"
        nome1 = (f"hlsp_tess-spoc_tess_phot_"
        f"{tic_str}-s0084_tess_v1_lc.fits")
        
        nome2 = (f"hlsp_tess-spoc_tess_phot_"
        f"{tic}-s0084_tess_v1_lc.fits")
        
        if nome1 in todos_arquivos:
            arquivos.append(todos_arquivos[nome1])
        
        elif nome2 in todos_arquivos:
            arquivos.append(todos_arquivos[nome2])
    
    print(f"\n{len(arquivos)} curvas de luz encontradas.")
    
    if len(arquivos) == 0:
        print("Nenhum FITS correspondente foi encontrado.")
    
    return arquivos

# Transformação igual do treino, mas sem labels e sem data augmentation
def infer_transform(observation, length=1400):
    try:
        if hasattr(observation, 'light_curve'):
            fluxes = observation.light_curve.fluxes
        else:
            fluxes = observation.fluxes
       
        # Tratamento de NaNs
        valid_mask = ~np.isnan(fluxes) & ~np.isinf(fluxes)
        mediana_bruta = np.median(fluxes[valid_mask]) if any(valid_mask) else 0
        fluxes = np.where(valid_mask, fluxes, mediana_bruta)

        #compressao 15x
        bin_factor = 15
        if len(fluxes) >= bin_factor:
            sobra = len(fluxes) % bin_factor
            if sobra != 0: fluxes = fluxes[:-sobra]
            fluxes = fluxes.reshape(-1, bin_factor).mean(axis=1)

        # Normalização com Scikit-learn
        if len(fluxes) > 0:
            scaler = StandardScaler()
            fluxes = scaler.fit_transform(fluxes.reshape(-1, 1)).flatten()

        if len(fluxes) == 0:
            fluxes_final = np.zeros(length, dtype=np.float32)
        elif len(fluxes) < length:
            pad_size = length - len(fluxes)
            fluxes_final = np.pad(fluxes, (0, pad_size), mode='constant', constant_values=0)
        else:
            fluxes_final = fluxes[:length]
   
        # Converte para Tensores PyTorch
        fluxes_native = np.ascontiguousarray(fluxes_final, dtype=np.float32)
        fluxes_tensor = torch.tensor(fluxes_native).unsqueeze(1)
       
        return fluxes_tensor
 
    except Exception as e:
        print(f"Erro na transformação de um arquivo de inferência: {e}")
        return torch.zeros((length, 1), dtype=torch.float32)
   
def main():

    print("Obtendo lista de arquivos para inferência...")
    paths = get_infer_paths()

    if len(paths) == 0:
        print("Nenhum arquivo encontrado. Encerrando.")
        return

    infer_light_curve_collection = LightCurveCollection.new(
        get_paths_function=lambda: paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path)

    test_light_curve_dataset = FiniteStandardLightCurveDataset.new(
        light_curve_collections=[infer_light_curve_collection],
        post_injection_transform=infer_transform)

    model = Hadryss.new(input_length=1400)
    device = get_device()
    model_path = os.path.join(caminho_raiz, "CBPF-FAPESP-Exoplanets-project", "exalted-flower-31_latest_model.pt")

    print(f"\nCarregando modelo:\n"
        f"{model_path}\n"
        f"Dispositivo: {device}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("\nIniciando inferência...")

    confidences = infer_session(infer_datasets=[test_light_curve_dataset],
        model=model,batch_size=300,device=device)[0]

    sorted_paths_with_confidences = sorted(zip(paths, confidences),
        key=lambda x: x[1], reverse=True)

    print("\nSalvando resultados...")

    resultados_tabela = []

    for path, confidence in sorted_paths_with_confidences:
        partes = path.name.split("_")

        tic_limpo = (partes[4].split("-")[0]
            if len(partes) > 4
            else path.name)

        resultados_tabela.append({
            "TIC_ID": tic_limpo,
            "Nome_do_Arquivo": path.name,
            "Confiança_Planeta": confidence,
            "Probabilidade_Planeta(%)": round(confidence * 100, 4)})

    df_resultados = pd.DataFrame(resultados_tabela)
    caminho_saida = os.path.join(caminho_raiz,"resultados_exoplanetas_s0084.csv")
    df_resultados.to_csv(caminho_saida, index=False)

    print(f"\nResultados salvos em:")
    print(caminho_saida)

if __name__ == "__main__":
    main()
