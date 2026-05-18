from qusi.model import Hadryss
from qusi.session import TrainHyperparameterConfiguration, train_session
from dataset import get_transit_train_dataset, get_transit_validation_dataset


def main():
    train_dataset = get_transit_train_dataset()
    validation_dataset = get_transit_validation_dataset()
    model = Hadryss.new(input_length=4000)

    hyperparameters = TrainHyperparameterConfiguration.new(batch_size=4,cycles=2,train_steps_per_cycle=5,validation_steps_per_cycle=2)

    train_session(
        train_datasets=[train_dataset],
        validation_datasets=[validation_dataset],
        model=model,
        hyperparameter_configuration=hyperparameters)


if __name__ == "__main__":
    main()