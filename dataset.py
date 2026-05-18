from pathlib import Path
from functools import partial
from astropy.io import fits
from qusi.data import (LightCurveDataset,LightCurveObservationCollection)
from qusi.transform import (default_light_curve_observation_post_injection_transform)

def get_positive_train_paths():
    return list(Path("dados/train/positives").glob("*.fits"))

def get_negative_train_paths():
    return list(Path("dados/train/negatives").glob("*.fits"))

def get_positive_validation_paths():
    return list(Path("dados/validation/positives").glob("*.fits"))

def get_negative_validation_paths():
    return list(Path("dados/validation/negatives").glob("*.fits"))

def get_positive_test_paths():
    return list(Path("dados/test/positives").glob("*.fits"))

def get_negative_test_paths():
    return list(Path("dados/test/negatives").glob("*.fits"))

#aqui precisei mudar pq as colunas dos dados nao sao as mesmas do qusi oriiginal (tutorial 3)
#def load_times_and_fluxes_from_path(path):
    #light_curve = TessMissionLightCurve.from_path(path)
    #eturn light_curve.times, light_curve.fluxes
def load_times_and_fluxes_from_path(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        times = data["TIME"]
        fluxes = data["FLUX"]

    return times, fluxes

#labels
def positive_label_function(path):
    return 1

def negative_label_function(path):
    return 0

#cortar curvas muito longas e repetir urvas muito curtas
train_transform = partial(
    default_light_curve_observation_post_injection_transform,length=4000,randomize=True)

validation_transform = partial(
    default_light_curve_observation_post_injection_transform,length=4000,randomize=False)

#train
def get_transit_train_dataset():
    positive_collection = LightCurveObservationCollection.new(
        get_paths_function=get_positive_train_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)

    negative_collection = LightCurveObservationCollection.new(
        get_paths_function=get_negative_train_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)

    dataset = LightCurveDataset.new(standard_light_curve_collections=[positive_collection,negative_collection],
post_injection_transform=train_transform)

    return dataset

#validation
def get_transit_validation_dataset():
    positive_collection = LightCurveObservationCollection.new(
        get_paths_function=get_positive_validation_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)

    negative_collection = LightCurveObservationCollection.new(
        get_paths_function=get_negative_validation_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)

    dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_collection,negative_collection],
        post_injection_transform=validation_transform)

    return dataset