import pandas as pd
import sys

def main():
    """
    Script per selezionare casualmente 1 riga 
    dal dataset corrupted HELOC DLLM
    Esclude RiskPerformance dal formato output
    """
    
    # Percorso del dataset
    dataset_path = 'data/processed/Fase2/DataCorruption/heloc_DLLM/heloc_DLLM_imputation_test_corrupted_MAR_10.csv'
    
    try:
        # Carica il dataset
        print(f"Caricamento dataset da: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Dataset caricato: {len(df)} righe totali, {len(df.columns)} colonne")
        
        # Seleziona casualmente 1 riga senza vincoli
        sample_row = df.sample(n=1, random_state=None)
        row_index = sample_row.index[0]
        row_data = sample_row.iloc[0]
        
        # Escludi RiskPerformance dal formato output
        filtered_data = row_data.drop('RiskPerformance', errors='ignore')
        
        # Output nel formato specificato
        print("\n" + "=" * 150)
        print(f"1 RIGA CASUALMENTE SELEZIONATA (indice: {row_index}, CSV line: {row_index + 1}):")
        print("=" * 150)
        
        # Formato: -NomeColonna=valore, -NomeColonna=valore, ...
        # Sostituisci nan con "???"
        formatted_values = [f"-{col}={'???' if pd.isna(val) else val}" for col, val in zip(filtered_data.index, filtered_data.values)]
        output_string = ", ".join(formatted_values) + "."
        print(output_string)
        
        print("=" * 150)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"ERRORE: File non trovato - {str(e)}")
        return 1
    except Exception as e:
        print(f"ERRORE: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
