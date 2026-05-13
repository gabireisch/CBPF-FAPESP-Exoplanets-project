import os
import random
import shutil
from pathlib import Path

# CONFIGURAÇÕES
random.seed(42)
VAL_PERCENT = 0.15
TEST_PERCENT = 0.15
BASE_DIR = "dados"

#divisao em train, validation e test
def split_dataset(class_name):
    source_dir = os.path.join(BASE_DIR, "train", class_name)
    val_dir = os.path.join(BASE_DIR, "validation", class_name)
    test_dir = os.path.join(BASE_DIR, "test", class_name)

    # cria pastas
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # lista arquivos
    files = list(Path(source_dir).glob("*.fits"))
    print(f"Classe: {class_name}")
    print(f"{len(files)} arquivos encontrados")

    # embaralha
    random.shuffle(files)

    # calcula quantidades
    n_total = len(files)
    n_val = int(n_total * VAL_PERCENT)
    n_test = int(n_total * TEST_PERCENT)

    # separa arquivos
    val_files = files[:n_val]
    test_files = files[n_val:n_val + n_test]
    print(f"Validation: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    print(f"Train restante: {n_total - n_val - n_test}")

    # move validation
    for file in val_files:
        destination = os.path.join(val_dir, file.name)
        shutil.move(str(file), destination)
        print(f"[VALIDATION] {file.name}")

    # move test
    for file in test_files:
        destination = os.path.join(test_dir, file.name)
        shutil.move(str(file), destination)
        print(f"[TEST] {file.name}")

split_dataset("positives")
split_dataset("negatives")
print("\nDivisão concluída!")