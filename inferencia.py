import sys
import os
from pathlib import Path
caminho_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(caminho_raiz)
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from qusi.data import FiniteStandardLightCurveDataset, LightCurveCollection
from qusi.model import Hadryss
from qusi.session import get_device, infer_session

# 2. Agora o Python acha o dataset indicando a pasta "scripts" antes!
from scripts.dataset import load_times_and_fluxes_from_path

def get_infer_paths():
    # Usa o caminho_raiz para encontrar a pasta exata, não importa de onde você rode o script
    caminho_dados = os.path.join(caminho_raiz, 'dados_inferencia_teste')
    arquivos = list(Path(caminho_dados).glob('*.fits'))
    
    if len(arquivos) == 0:
        print(f"Nenhum arquivo .fits encontrado na pasta {caminho_dados}")
    else:
        print(f"{len(arquivos)} curvas de luz carregadas para inferência.")
        
    return arquivos

#Transformação igual do treino, mas sem labels e sem data augmentation
def infer_transform(observation, length=1400):
    try: 
        # Dependendo da versão do qusi na inferência, o objeto pode vir encapsulado
        if hasattr(observation, 'light_curve'):
            fluxes = observation.light_curve.fluxes
        else:
            fluxes = observation.fluxes
        
        #Tratamento de NaNs 
        valid_mask = ~np.isnan(fluxes) & ~np.isinf(fluxes)
        mediana_bruta = np.median(fluxes[valid_mask]) if any(valid_mask) else 0
        fluxes = np.where(valid_mask, fluxes, mediana_bruta)

        #Binning 15x
        bin_factor = 15
        if len(fluxes) >= bin_factor:
            sobra = len(fluxes) % bin_factor
            if sobra != 0: fluxes = fluxes[:-sobra]
            fluxes = fluxes.reshape(-1, bin_factor).mean(axis=1)

        #Normalização com Scikit-learn 
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
        
        # Retorna apenas o tensor da curva puramente
        return fluxes_tensor
  
    except Exception as e:
        print(f"Erro na transformação de um arquivo de inferência: {e}")
        # Retorna apenas o tensor vazio
        return torch.zeros((length, 1), dtype=torch.float32)
    
def main():
    infer_light_curve_collection = LightCurveCollection.new(
        get_paths_function=get_infer_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path)

    test_light_curve_dataset = FiniteStandardLightCurveDataset.new(
       light_curve_collections=[infer_light_curve_collection],
       post_injection_transform=infer_transform)

    #tamanho da entrada
    model = Hadryss.new(input_length=1400)
    device = get_device()
    model_path = 'sessions/exalted-flower-31_latest_model.pt' #nome do modelo laranja
    print(f"Carregando a Rede Neural: {model_path} no dispositivo {device}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("Processando inferência nas novas curvas de luz...")
    confidences = infer_session(
        infer_datasets=[test_light_curve_dataset], 
        model=model,
        batch_size=100, 
        device=device)[0]
        
    paths = list(get_infer_paths())
    paths_with_confidences = zip(paths, confidences)
    
    # Ordena os arquivos da maior chance de planeta para a menor
    sorted_paths_with_confidences = sorted(
        paths_with_confidences, key=lambda path_with_confidence: path_with_confidence[1], reverse=True)
        
    print("RESULTADOS")
    for path, confidence in sorted_paths_with_confidences:
        print(f'{path.name:<25} -> {confidence * 100:>6.2f}% de chance de trânsito')

if __name__ == '__main__':
    main()