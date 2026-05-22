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
        #path como string p evitar erros silenciosos no lightkurve
        lc = read(str(path))
        return lc.time.value, lc.flux.value
    except Exception as e:
        print(f"Erro ao ler arquivo {path}: {e}")
        raise e  # Força o programa a parar e mostrar o erro em vez de entrar em loop

def positive_label_function(path): 
    return 1
def negative_label_function(path): 
    return 0

#transformação customizada que faz exatamente o que o qusi faria, mas sem conflitos de versão e já retornando os Tensores PyTorch corretos.
def custom_transform(observation, length=2000, randomize=False):
    try:
        # Extrai os dados do objeto Observation
        fluxes = observation.light_curve.fluxes
        label = observation.label

        #limpa nans e infs
        valid_mask = ~np.isnan(fluxes) & ~np.isinf(fluxes)
        fluxes = fluxes[valid_mask]

        #randomização (corte e rolagem aleatória, apenas no treino)
        if randomize and len(fluxes) > 0:
            roll_amount = np.random.randint(0, len(fluxes))
            fluxes = np.roll(fluxes, roll_amount)

        #tamanho (2000)
        if len(fluxes) == 0:
            fluxes = np.zeros(length, dtype=np.float32)
        elif len(fluxes) < length:
            repeats = (length // len(fluxes)) + 1
            fluxes = np.tile(fluxes, repeats)[:length]
        else:
            fluxes = fluxes[:length]

        #normalização
        median = np.median(fluxes)
        mad = np.median(np.abs(fluxes - median))
        if mad == 0: 
            mad = 1e-8
        fluxes = (fluxes - median) / mad

        #converte para Tensores PyTorch (o modelo Hadryss exige shape [length, 1])
        fluxes_tensor = torch.tensor(fluxes, dtype=torch.float32).unsqueeze(1)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return fluxes_tensor, label_tensor
    
    except Exception as e:
        print(f"Erro na transformação dos dados: {e}")
        raise e  # Força o erro a aparecer na tela

train_transform = partial(custom_transform, length=2000, randomize=True)
val_test_transform = partial(custom_transform, length=2000, randomize=False)

def get_transit_train_dataset():
    positive_train_collection = LightCurveObservationCollection.new(get_paths_function=get_positive_train_paths,load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    
    negative_train_collection = LightCurveObservationCollection.new(get_paths_function=get_negative_train_paths,load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)
    
    train_dataset = LightCurveDataset.new(standard_light_curve_collections=[positive_train_collection, negative_train_collection],
        post_injection_transform=train_transform)
    return train_dataset

def get_transit_validation_dataset():
    positive_validation_collection = LightCurveObservationCollection.new(get_paths_function=get_positive_validation_paths,load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    
    negative_validation_collection = LightCurveObservationCollection.new(get_paths_function=get_negative_validation_paths, load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)
    
    validation_dataset = LightCurveDataset.new(standard_light_curve_collections=[positive_validation_collection, negative_validation_collection],
        post_injection_transform=val_test_transform)
    return validation_dataset

def get_transit_test_dataset():
    positive_test_collection = LightCurveObservationCollection.new(get_paths_function=get_positive_test_paths,load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    
    negative_test_collection = LightCurveObservationCollection.new(get_paths_function=get_negative_test_paths, load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)
    
    test_dataset = LightCurveDataset.new(standard_light_curve_collections=[positive_test_collection, negative_test_collection],
        post_injection_transform=val_test_transform)
    return test_dataset