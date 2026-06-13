import pandas as pd
import numpy as np
import os

def calculate_metrics(gt_path, imp_path, mask_path):
    print(f"Caricamento dati...")
    df_gt = pd.read_csv(gt_path)
    df_imp = pd.read_csv(imp_path)
    df_mask = pd.read_csv(mask_path)
    
    if not (df_gt.shape == df_imp.shape == df_mask.shape):
        raise ValueError(f"Le dimensioni non coincidono: GT {df_gt.shape}, Imp {df_imp.shape}, Mask {df_mask.shape}")
        
    mae_dict = {}
    mse_dict = {}
    acc_dict = {}
    
    all_numeric_errors = []
    
    for col in df_gt.columns:
        # Prendi la maschera booleana per questa colonna (True se il valore era mancante originariamente)
        col_mask = df_mask[col].astype(bool).values
        
        if col_mask.sum() == 0:
            continue
            
        # Proviamo a convertire in numerico
        gt_numeric = pd.to_numeric(df_gt[col], errors='coerce')
        imp_numeric = pd.to_numeric(df_imp[col], errors='coerce')
        
        # Se la colonna ha molti NaN dopo la conversione ed è di tipo object, 
        # potrebbe essere una variabile categorica (es. RiskPerformance)
        if gt_numeric.isna().sum() == len(df_gt) and df_gt[col].dtype == 'object':
            # Calcolo Accuracy
            gt_vals = df_gt[col].values[col_mask]
            imp_vals = df_imp[col].values[col_mask]
            correct = (gt_vals == imp_vals).sum()
            acc_dict[col] = correct / len(gt_vals)
            continue
            
        # Variabile numerica
        gt_vals = gt_numeric.values[col_mask]
        imp_vals = imp_numeric.values[col_mask]
        
        # Ignora i casi in cui il dato originale non era un vero numero (es. 'Never Had Delinquency')
        # perchè pd.to_numeric lo ha convertito in NaN
        valid_mask = ~np.isnan(gt_vals) & ~np.isnan(imp_vals)
        valid_gt = gt_vals[valid_mask]
        valid_imp = imp_vals[valid_mask]
        
        if len(valid_gt) > 0:
            errors = valid_gt - valid_imp
            all_numeric_errors.extend(errors)
            
            mae_dict[col] = np.mean(np.abs(errors))
            mse_dict[col] = np.mean(errors**2)
            
    print("\n--- METRICHE PER COLONNA ---")
    for col in mae_dict:
        print(f"Colonna: {col:35} | MAE: {mae_dict[col]:.4f} | MSE: {mse_dict[col]:.4f}")
        
    for col in acc_dict:
        print(f"Colonna: {col:35} | Accuracy: {acc_dict[col]:.4f}")
        
    print("\n--- METRICHE GLOBALI (valori numerici) ---")
    if len(all_numeric_errors) > 0:
        all_numeric_errors = np.array(all_numeric_errors)
        overall_mae = np.mean(np.abs(all_numeric_errors))
        overall_mse = np.mean(all_numeric_errors**2)
        print(f"OVERALL MAE: {overall_mae:.4f}")
        print(f"OVERALL MSE: {overall_mse:.4f}")
    else:
        print("Nessun valore numerico valido trovato per il calcolo.")

if __name__ == "__main__":
    # Path dei file
    base_dir_gt = "/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase2/SplitDataset/Split_DLLM"
    base_dir_mask = "/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase2/DataCorruption/heloc_DLLM"
    base_dir_imp = "/Users/marco/Documents/Biometry/DLL/DLLM_Project/DLLM_Project/data/processed/Fase3/Imputated_DLLM"
    
    gt_file = os.path.join(base_dir_gt, "heloc_DLLM_imputation_test.csv")
    mask_file = os.path.join(base_dir_mask, "heloc_DLLM_imputation_test_mask_MCAR_10.csv")
    imp_file = os.path.join(base_dir_imp, "heloc_DLLM_discriminative_train_MCAR_10.csv")
    
    calculate_metrics(gt_file, imp_file, mask_file)
