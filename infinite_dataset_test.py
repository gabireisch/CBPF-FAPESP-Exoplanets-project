import torch
from qusi.model import Hadryss
from qusi.session import get_device, infinite_datasets_test_session
from torch.nn import BCELoss
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

# 1. Importamos diretamente a função pronta do seu dataset.py
from dataset import get_transit_test_dataset 

def main():
    # 2. Carregamos o dataset com apenas uma linha
    test_light_curve_dataset = get_transit_test_dataset()

    model = Hadryss.new(input_length=2000)
    device = get_device()
    
    # 3. Substituí o <wandb_run_name> pelo nome real que você usou no seu outro script (old-force-1)
    model.load_state_dict(torch.load('sessions/old-force-1_latest_model.pt', map_location=device))
    
    metric_functions = [BinaryAccuracy(), BCELoss(), BinaryAUROC()]
    results = infinite_datasets_test_session(
        test_datasets=[test_light_curve_dataset], 
        model=model,
        metric_functions=metric_functions, 
        batch_size=100, 
        device=device,
        steps=100
    )
    return results

if __name__ == '__main__':
    main()