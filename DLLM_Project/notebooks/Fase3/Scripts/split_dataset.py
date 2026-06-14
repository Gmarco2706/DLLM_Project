import pandas as pd
import numpy as np
import os

def split_dataset(input_path, num_splits=6):
    print(f"Loading dataset from:\n  {input_path}")
    df = pd.read_csv(input_path)
    
    # Divide l'indice del dataframe in `num_splits` parti, sequenzialmente.
    # array_split gestisce automaticamente il caso in cui il numero di righe 
    # non sia perfettamente divisibile per num_splits, creando parti quasi uguali.
    splits_indices = np.array_split(df.index, num_splits)
    splits = [df.loc[idx] for idx in splits_indices]
    
    output_files = []
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    print(f"\nDividendo in {num_splits} parti...")
    for i, split_df in enumerate(splits):
        # Aggiunge un suffisso con il numero della parte
        output_path = os.path.join(base_dir, f"{base_name}_part{i+1}.csv")
        
        # Salva senza indice per preservare l'esatta struttura originale
        split_df.to_csv(output_path, index=False)
        output_files.append(output_path)
        print(f"Parte {i+1} salvata ({len(split_df)} righe): {output_path}")
 
    return output_files

if __name__ == "__main__":
    input_csv = "/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase2/DataCorruption/heloc_DLLM/heloc_DLLM_imputation_test_corrupted_MNAR_25.csv"
    
    # Esegue lo split
    split_dataset(input_csv, num_splits=6)
