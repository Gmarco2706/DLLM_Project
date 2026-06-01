import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys

def main():
    """
    Script per selezionare 2 righe GOOD e 2 righe BAD vettorialmente SIMILI tra di loro
    dal dataset HELOC che contengono solo valori numerici (senza stringhe di dati mancanti).
    
    Metrica di similarità: Euclidean Distance (trova le coppie con distanza minima)
    """
    
    # Configurazione
    dataset_path = 'data/processed/DatasetFinali/heloc_dataset_cleanedDLLM.csv'
    
    # Stringhe che indicano dati mancanti
    missing_strings = [
        'No Installment Burden',
        'Never Had Delinquency',
        'No Installment Loans',
        'No Bank Revolving',
        'No Revolving Burden',
        'No Revolving Accounts',
        'No Payment History',
        'No Inquiry Record',
        'No Credit Inquiry',
        'New to Credit System'
    ]
    
    try:
        # Carica il dataset
        print(f"Caricamento dataset da: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Dataset caricato: {len(df)} righe totali")
        
        # Funzione per verificare se una riga contiene solo valori numerici
        def has_only_numeric_values(row):
            row_str = str(row).lower()
            for s in missing_strings:
                if s.lower() in row_str:
                    return False
            return True
        
        # Applica il filtro
        df['is_numeric_only'] = df.apply(has_only_numeric_values, axis=1)
        
        # Separa Good e Bad
        good_rows = df[(df['RiskPerformance'] == 'Good') & (df['is_numeric_only'] == True)]
        bad_rows = df[(df['RiskPerformance'] == 'Bad') & (df['is_numeric_only'] == True)]
        
        print(f"\nRighe disponibili con solo valori numerici:")
        print(f"  - GOOD: {len(good_rows)}")
        print(f"  - BAD: {len(bad_rows)}")
        
        # Verifica che ci siano abbastanza righe
        if len(good_rows) < 2 or len(bad_rows) < 2:
            print("\nERRORE: Non ci sono abbastanza righe per la selezione!")
            return 1
        
        # Funzione per trovare la coppia vettorialmente più simile usando Euclidean distance
        def find_similar_pair(data, n_samples=500):
            """
            Trova la coppia di righe più simile basato su Euclidean distance.
            Se ci sono troppi dati, campiona casualmente.
            """
            print(f"  Ricerca della coppia più simile tra {len(data)} righe...")
            
            # Se ci sono troppi dati, sampliamo casualmente per performance
            if len(data) > n_samples:
                sampled_indices = np.random.choice(len(data), n_samples, replace=False)
                data_sample = data.iloc[sampled_indices].reset_index(drop=True)
                original_indices = data.index[sampled_indices]
            else:
                data_sample = data.reset_index(drop=True)
                original_indices = data.index
            
            # Estrai solo le colonne numeriche (escludi RiskPerformance e is_numeric_only)
            numeric_cols = data_sample.select_dtypes(include=[np.number]).columns.tolist()
            X = data_sample[numeric_cols].values
            
            # Normalizza i valori per fare un confronto equo
            X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            # Calcola la distanza euclidea tra tutte le coppie
            distances = euclidean_distances(X_normalized)
            
            # Mascherizza la diagonale (distanza da sé stesso)
            np.fill_diagonal(distances, np.inf)
            
            # Trova la coppia con la distanza minima
            min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
            min_distance = distances[min_distance_idx]
            
            # Ritorna gli indici originali e la distanza
            idx1 = original_indices[min_distance_idx[0]]
            idx2 = original_indices[min_distance_idx[1]]
            
            return data.loc[idx1], data.loc[idx2], idx1, idx2, min_distance
        
        # Trova coppie simili per GOOD e BAD
        print("\nRicerca coppie simili GOOD...")
        good_row1, good_row2, good_idx1, good_idx2, good_dist = find_similar_pair(good_rows)
        
        print("Ricerca coppie simili BAD...")
        bad_row1, bad_row2, bad_idx1, bad_idx2, bad_dist = find_similar_pair(bad_rows)
        
        # Output
        print("\n" + "=" * 140)
        print("2 RIGHE GOOD VETTORIALMENTE SIMILI (distanza euclidea normalizzata):")
        print("=" * 140)
        print(f"Distanza tra le righe: {good_dist:.4f}\n")
        
        for idx, (i, row) in enumerate([(good_idx1, good_row1), (good_idx2, good_row2)], 1):
            csv_line = i + 1
            print(f"Riga {idx} GOOD (indice dataset: {i}, CSV line: {csv_line}):")
            # Formato: -NomeColonna=valore, escludi is_numeric_only
            formatted_values = [f"-{col}={val}" for col, val in zip(row.index, row.values) 
                              if col not in ['is_numeric_only']]
            print(", ".join(formatted_values) + ".\n")
        
        print("\n" + "=" * 140)
        print("2 RIGHE BAD VETTORIALMENTE SIMILI (distanza euclidea normalizzata):")
        print("=" * 140)
        print(f"Distanza tra le righe: {bad_dist:.4f}\n")
        
        for idx, (i, row) in enumerate([(bad_idx1, bad_row1), (bad_idx2, bad_row2)], 1):
            csv_line = i + 1
            print(f"Riga {idx} BAD (indice dataset: {i}, CSV line: {csv_line}):")
            # Formato: -NomeColonna=valore, escludi is_numeric_only
            formatted_values = [f"-{col}={val}" for col, val in zip(row.index, row.values) 
                              if col not in ['is_numeric_only']]
            print(", ".join(formatted_values) + ".\n")
        
        print("=" * 140)
        
        return 0
        
    except FileNotFoundError:
        print(f"ERRORE: Dataset non trovato in {dataset_path}")
        return 1
    except Exception as e:
        print(f"ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
