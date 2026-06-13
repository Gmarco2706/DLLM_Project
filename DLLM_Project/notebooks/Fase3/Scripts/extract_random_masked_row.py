import os
import random
import pandas as pd
from pathlib import Path

def extract_random_masked_row():
    # Definisce il percorso alla cartella dei dataset corrotti della pipeline DLLM
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / 'data' / 'processed' / 'Fase2' / 'DataCorruption' / 'heloc_DLLM'
    
    if not data_dir.exists():
        raise FileNotFoundError(f"La cartella {data_dir} non esiste. Assicurati di eseguire lo script dalla root del progetto.")
    
    # Trova tutti i dataset mascherati (corrupted)
    corrupted_files = list(data_dir.glob("*_corrupted_*.csv"))
    
    if not corrupted_files:
        raise ValueError(f"Nessun file 'corrupted' trovato in {data_dir}")
    
    # Sceglie randomicamente un file
    random_file = random.choice(corrupted_files)
    print(f"Dataset selezionato: {random_file.name}")
    
    # Carica il dataset
    df = pd.read_csv(random_file)
    
    # Filtra le righe che contengono almeno un valore mancante (NaN), che indicano la mascheratura
    masked_rows = df[df.isnull().any(axis=1)]
    
    if masked_rows.empty:
        raise ValueError(f"Nessuna riga mascherata trovata nel file {random_file.name}")
    
    # Seleziona randomicamente una riga tra quelle mascherate
    random_row = masked_rows.sample(n=1)
    
    # Restituisce la riga come dizionario per facilitarne l'ispezione o il passaggio al modello
    row_dict = random_row.to_dict(orient='records')[0]
    
    print("\n--- Riga mascherata estratta ---")
    output_parts = []
    for col, val in row_dict.items():
        if pd.isna(val):
            output_parts.append(f"- {col}: ???")
        else:
            output_parts.append(f"- {col}: {val}")
            
    print(" ".join(output_parts))
            
    return row_dict

if __name__ == "__main__":
    extract_random_masked_row()
