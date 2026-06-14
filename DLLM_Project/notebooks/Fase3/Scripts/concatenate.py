import pandas as pd
import os

# Lista dei file generati nello stesso ordine originale
files = [
    '/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase3/Imputated_DLLM/DiscriminativeParts/heloc_DLLM_discriminative_train_MNAR_25_part1.csv',
    '/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase3/Imputated_DLLM/DiscriminativeParts/heloc_DLLM_discriminative_train_MNAR_25_part2.csv',
    '/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase3/Imputated_DLLM/DiscriminativeParts/heloc_DLLM_discriminative_train_MNAR_25_part3.csv',
    '/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase3/Imputated_DLLM/DiscriminativeParts/heloc_DLLM_discriminative_train_MNAR_25_part4.csv',
    '/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase3/Imputated_DLLM/DiscriminativeParts/heloc_DLLM_discriminative_train_MNAR_25_part5.csv',
    '/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase3/Imputated_DLLM/DiscriminativeParts/heloc_DLLM_discriminative_train_MNAR_25_part6.csv',
]

# Concatena tutte le parti
combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Salva il file combinato finale
combined_df.to_csv('/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase3/Imputated_DLLM/heloc_DLLM_discriminative_train_MNAR_25.csv', index=False)

