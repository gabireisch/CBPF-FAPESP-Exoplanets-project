import torch

from qusi.data import LightCurveDataset
from qusi.data import LightCurveObservationCollection
from qusi.model import Hadryss
from qusi.session import get_device, infinite_datasets_test_session
from torch.nn import BCELoss
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from dataset import get_positive_test_paths, load_times_and_fluxes_from_path, positive_label_function, \
    get_negative_test_paths, negative_label_function


def main():
    positive_test_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_positive_test_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=positive_label_function)
    negative_test_light_curve_collection = LightCurveObservationCollection.new(
        get_paths_function=get_negative_test_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path,
        load_label_from_path_function=negative_label_function)

    test_light_curve_dataset = LightCurveDataset.new(
        standard_light_curve_collections=[positive_test_light_curve_collection,
                                          negative_test_light_curve_collection])

    model = Hadryss.new()
    device = get_device()
    model.load_state_dict(torch.load('sessions/<wandb_run_name>_latest_model.pt', map_location=device))
    metric_functions = [BinaryAccuracy(), BCELoss(), BinaryAUROC()]
    results = infinite_datasets_test_session(test_datasets=[test_light_curve_dataset], model=model,
                                             metric_functions=metric_functions, batch_size=100, device=device,
                                             steps=100)
    return results


if __name__ == '__main__':
    main()
