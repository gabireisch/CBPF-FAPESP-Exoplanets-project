from pathlib import Path
from lightkurve import read
from functools import partial
import numpy as np
import torch
from qusi.data import LightCurveDataset, LightCurveObservationCollection

def get_positive_train_paths(): 
    return list(Path('dados/train/positives').glob('*.fits'))
def get_negative_train_paths(): 
    return list(Path('dados/train/negatives').glob('*.fits'))
def get_positive_validation_paths(): 
    return list(Path('dados/validation/positives').glob('*.fits'))
def get_negative_validation_paths(): 
    return list(Path('dados/validation/negatives').glob('*.fits'))
def get_positive_test_paths(): 
    return list(Path('dados/test/positives').glob('*.fits'))
def get_negative_test_paths(): 
    return list(Path('dados/test/negatives').glob('*.fits'))

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
        valid_mask = ~np.isnan(fluxes) & ~np.isinf(fluxes)
        fluxes = fluxes[valid_mask]

        # Binning: Converte cadência de 2 min para 30 min
        bin_factor = 15
        if len(fluxes) >= bin_factor:
            sobra = len(fluxes) % bin_factor
            if sobra != 0:
                fluxes = fluxes[:-sobra]
            fluxes = fluxes.reshape(-1, bin_factor).mean(axis=1)

        if randomize and len(fluxes) > 0:
            roll_amount = np.random.randint(0, len(fluxes))
            fluxes = np.roll(fluxes, roll_amount)

        # Padronização de tamanho pós-binning
        if len(fluxes) == 0:
            fluxes = np.zeros(length, dtype=np.float32)
        elif len(fluxes) < length:
            repeats = (length // len(fluxes)) + 1
            fluxes = np.tile(fluxes, repeats)[:length]
        else:
            fluxes = fluxes[:length]

        # Normalização 
        median = np.median(fluxes)
        mad = np.median(np.abs(fluxes - median))
        if mad == 0: mad = 1e-8
        fluxes = (fluxes - median) / mad
    
        fluxes_native = np.ascontiguousarray(fluxes, dtype=np.float32)
        fluxes_tensor = torch.tensor(fluxes_native).unsqueeze(1)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return fluxes_tensor, label_tensor
  
    except Exception as e:
        print(f"Erro na transformação dos dados: {e}")
        raise e

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