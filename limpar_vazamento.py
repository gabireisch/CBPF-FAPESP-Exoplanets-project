import os
import glob
from pathlib import Path

BASE_DIR = "dados_multisetor"
classes = ["positives", "negatives"]

print(f"Iniciando a varredura contra vazamentos de dados na pasta '{BASE_DIR}'...\n")

arquivos_deletados = 0

for classe in classes:
    print(f"Analisando classe: {classe.upper()}")
    
    # 1. Pega os caminhos de todos os arquivos
    train_files = glob.glob(os.path.join(BASE_DIR, "train", classe, "*.fits"))
    val_files = glob.glob(os.path.join(BASE_DIR, "validation", classe, "*.fits"))
    test_files = glob.glob(os.path.join(BASE_DIR, "test", classe, "*.fits"))
    
    # Função para extrair o TIC ID do nome do arquivo (ex: "TIC_123_sec4.fits" -> "123")
    def extrair_tic(caminho_arquivo):
        nome_arquivo = os.path.basename(caminho_arquivo)
        return nome_arquivo.split('_')[1]

    # 2. Mapeia quais TICs estão no Treino e na Validação
    tics_no_treino = set([extrair_tic(f) for f in train_files])
    tics_na_validacao = set([extrair_tic(f) for f in val_files])
    
    # 3. Limpa a pasta VALIDATION (Se o TIC já está no Treino, deleta da Validação)
    for f in val_files:
        tic = extrair_tic(f)
        if tic in tics_no_treino:
            os.remove(f)
            arquivos_deletados += 1
            print(f"{os.path.basename(f)} deletado do Validation (Já existe no Train)")
            tics_na_validacao.discard(tic) # Remove do set local já que apagamos o arquivo
            
    # 4. Limpa a pasta TEST (Se o TIC está no Treino OU na Validação, deleta do Test)
    for f in test_files:
        tic = extrair_tic(f)
        if tic in tics_no_treino or tic in tics_na_validacao:
            os.remove(f)
            arquivos_deletados += 1
            print(f"[VAZAMENTO CORRIGIDO] {os.path.basename(f)} deletado do Test (Já existe em outra pasta)")

print(f"Total de arquivos duplicados deletados: {arquivos_deletados}")