from pathlib import Path
from lightkurve import read
from functools import partial
import numpy as np
import torch
from qusi.data import LightCurveDataset, LightCurveObservationCollection
from sklearn.preprocessing import StandardScaler 

def get_positive_train_paths(): 
    return list(Path('dados_multisetor/train/positives').glob('*.fits'))
def get_negative_train_paths(): 
    return list(Path('dados_multisetor/train/negatives').glob('*.fits'))
def get_positive_validation_paths(): 
    return list(Path('dados_multisetor/validation/positives').glob('*.fits'))
def get_negative_validation_paths(): 
    return list(Path('dados_multisetor/validation/negatives').glob('*.fits'))
def get_positive_test_paths(): 
    return list(Path('dados_multisetor/test/positives').glob('*.fits'))
def get_negative_test_paths(): 
    return list(Path('dados_multisetor/test/negatives').glob('*.fits'))

def load_times_and_fluxes_from_path(path):
    try:
        lc = read(str(path))
        return lc.time.value, lc.flux.value
    except Exception as e:
        print(f"Erro ao ler arquivo {path}: {e}")
        raise e

def positive_label_function(path): 
    return 1
def negative_label_function(path): 
    return 0

def custom_transform(observation, length=1400, randomize=False):
    try: 
        fluxes = observation.light_curve.fluxes
        label = observation.label
        
        #NaNs
        valid_mask = ~np.isnan(fluxes) & ~np.isinf(fluxes)
        mediana_bruta = np.median(fluxes[valid_mask]) if any(valid_mask) else 0
        fluxes = np.where(valid_mask, fluxes, mediana_bruta)

        #Binning (Compressão de 15x -> Cadência de 30 min) (média de 15 pontos = 1 ponto)
        bin_factor = 15
        if len(fluxes) >= bin_factor:
            sobra = len(fluxes) % bin_factor
            if sobra != 0: fluxes = fluxes[:-sobra]
            fluxes = fluxes.reshape(-1, bin_factor).mean(axis=1)

        # Alterações para Data Augmentation (aplicado só no treino)

        #Tirar pontos aleatórios (apaga 10% dos dados)
        if randomize and len(fluxes) > 0:
            
            #Dropout: Tira pontos aleatórios (apaga 10% dos dados)
            dropout_mask = np.random.rand(len(fluxes)) > 0.10 
            fluxes = fluxes[dropout_mask]

            #Roll: Corta a curva e muda de lugar (Shift no tempo)
            roll_amount = np.random.randint(0, len(fluxes))
            fluxes = np.roll(fluxes, roll_amount)

        # Normalização com Scikit-learn (Média dividida pelo Desvio Padrão)
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
    
        #Converte para Tensores PyTorch
        fluxes_native = np.ascontiguousarray(fluxes_final, dtype=np.float32)
        fluxes_tensor = torch.tensor(fluxes_native).unsqueeze(1)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return fluxes_tensor, label_tensor
  
    except Exception as e:
        print(f"Erro na transformação dos dados: {e}")
        # Retorna tensor vazio para não travar o loop de treino
        return torch.zeros(length, 1), torch.tensor(observation.label, dtype=torch.float32)

train_transform = partial(custom_transform, length=1400, randomize=True)
val_test_transform = partial(custom_transform, length=1400, randomize=False)

def get_transit_train_dataset():
    pos = LightCurveObservationCollection.new(get_positive_train_paths, load_times_and_fluxes_from_path, positive_label_function)
    neg = LightCurveObservationCollection.new(get_negative_train_paths, load_times_and_fluxes_from_path, negative_label_function)
    return LightCurveDataset.new([pos, neg], post_injection_transform=train_transform)

def get_transit_validation_dataset():
    pos = LightCurveObservationCollection.new(get_positive_validation_paths, load_times_and_fluxes_from_path, positive_label_function)
    neg = LightCurveObservationCollection.new(get_negative_validation_paths, load_times_and_fluxes_from_path, negative_label_function)
    return LightCurveDataset.new([pos, neg], post_injection_transform=val_test_transform)

def get_transit_test_dataset():
    pos = LightCurveObservationCollection.new(get_positive_test_paths, load_times_and_fluxes_from_path, positive_label_function)
    neg = LightCurveObservationCollection.new(get_negative_test_paths, load_times_and_fluxes_from_path, negative_label_function)
    return LightCurveDataset.new([pos, neg], post_injection_transform=val_test_transform)