from qusi.model import Hadryss
from qusi.session import TrainHyperparameterConfiguration, train_session
from dataset import get_transit_train_dataset, get_transit_validation_dataset

def main():
    train_light_curve_dataset = get_transit_train_dataset()
    validation_light_curve_dataset = get_transit_validation_dataset()
    
    # entrada tamanho 1400
    model = Hadryss.new(input_length= 1400)
    
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(batch_size=64, cycles=50,  
    train_steps_per_cycle=25,       # 1594 fits / 64 = 24.9
    validation_steps_per_cycle=6)   # 360 fits / 64 = 5.6)
    
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
         model=model, hyperparameter_configuration=train_hyperparameter_configuration)

if __name__ == '__main__':
    main()