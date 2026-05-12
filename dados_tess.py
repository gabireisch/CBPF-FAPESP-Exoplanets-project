import pandas as pd


def filtrar_tic_ids(csv_path):

    # lê o CSV
    df = pd.read_csv(csv_path,sep=None,engine="python")
    df.columns = df.columns.str.strip()

    tic_col = df.columns[1] # segunda coluna - TIC ID
    disp_col = df.columns[2] #terceira coluna - caracterizacao 

    print(f"TIC column: {tic_col}")
    print(f"Disposition column: {disp_col}")

    df[disp_col] = df[disp_col].astype(str).str.strip()

    # positivos = Confirmed Planet + Known Planet
    positivos = df[df[disp_col].isin(["CP", "KP"])][[tic_col]]

    # negativos = False Positive + False Alarm
    negativos = df[df[disp_col].isin(["FA", "FP"])][[tic_col]]

    # candidatos = Planetary Candidate
    candidatos = df[df[disp_col] == "PC"][[tic_col]]

    # ambiguos
    ambiguos = df[df[disp_col] == "APC"][[tic_col]]

    print(f"\nPositivos: {len(positivos)}")
    print(f"Negativos: {len(negativos)}")
    print(f"Candidatos: {len(candidatos)}")
    print(f"Ambíguos   (APC):     {len(ambiguos)}")

    return positivos, negativos, candidatos, ambiguos

positivos, negativos, candidatos, ambiguos = filtrar_tic_ids("dadosTESS.csv")

# salva CSVs separados
positivos.to_csv("positivos.csv",index=False)
negativos.to_csv("negativos.csv",index=False)
candidatos.to_csv("candidatos.csv",index=False)
ambiguos.to_csv("ambiguos.csv",index=False)

print("\nArquivos salvos.")