# HELOC Dataset Preprocessing

## Overview

Questo documento descrive il processo di preprocessing del dataset HELOC per prepararlo al training di modelli DLLM.

## Pipeline di Preprocessing

Il pipeline è strutturato in step progressivi, consentendo a chi utilizza la repo di fermarsi a qualsiasi stadio intermedio se necessario:

1. **Dataset originale** → `data/raw/heloc_dataset.csv` (10,459 righe)
2. **Dopo rimozione -9** → `data/processed/heloc_dataset_no9.csv` (10,371 righe)
3. **Dopo sostituzione -7** → `data/processed/heloc_dataset_no9_no7.csv` (10,371 righe, 2 col mixed-type)
4. **Finale (completo)** → `data/processed/heloc_dataset_cleaned.csv` (9,861 righe)

## Special Values Handling

### Valore -9: Missing/Insufficient Data
- **Azione**: Rimozione completa della riga
- **Righe rimosse**: 588 (5.62% del dataset originale)
- **Giustificazione**: Dati insufficienti per l'addestramento
- **File output**: `data/processed/heloc_dataset_no9.csv`

### Valore -7: Insufficient/Account Not Active
- **Azione**: Sostituzione con stringhe semantiche
- **Colonne interessate**: 2
- **Total valori**: 206
- **File output**: `data/processed/heloc_dataset_no9_no7.csv`

| Colonna | Valore Semantico | Count |
|---------|-------------------|-------|
| MSinceMostRecentDelq | "Never Had Delinquency" | 103 |
| MSinceMostRecentInqexcl7days | "No Credit Inquiry" | 103 |

**Nota tecnica**: Dopo la sostituzione di -7, le colonne MSinceMostRecentDelq e MSinceMostRecentInqexcl7days diventano di tipo misto (numeric + string), causando una conversione automatica a object dtype al salvataggio/ricaricamento CSV.

### Valore -8: Applicant Too Young/Account Not Active
- **Azione**: Sostituzione con stringhe semantiche
- **Colonne interessate**: 9
- **Total valori**: 11,072
- **File output**: `data/processed/heloc_dataset_cleaned.csv`

| Colonna | Valore Semantico | Count |
|---------|-------------------|-------|
| MSinceOldestTradeOpen | "New to Credit System" | 239 |
| PercentTradesWBalance | "No Valid Credit Accounts" | 11 |
| NumRevolvingTradesWBalance | "No Revolving Accounts" | 149 |
| NumBank2NatlTradesWHighUtilization | "No Bank Revolving" | 576 |
| NetFractionRevolvingBurden | "No Revolving Burden" | 179 |
| NumInstallTradesWBalance | "No Installment Loans" | 854 |
| NetFractionInstallBurden | "No Installment Burden" | 3,412 |
| MSinceMostRecentDelq | "No Payment History" | 176 |
| MSinceMostRecentInqexcl7days | "No Inquiry Record" | 476 |

**Nota tecnica**: La sostituzione di -8 in MSinceMostRecentDelq e MSinceMostRecentInqexcl7days utilizza mapping con chiavi string `{"-8": "..."}` a causa della conversione di tipo avvenuta nello step precedente, mentre le altre colonne usano chiavi numeric `{-8: "..."}`.

## Final Dataset

- **Righe**: 9,861 (da 10,459 originali, -598 = -5.72%)
- **Colonne**: 24 (invariate)
- **Special values rimasti**: 0

## Implementation Notes

- **Framework**: Python 3.x con pandas
- **Metodo sostituzione**: `pandas.DataFrame.replace()`
- **Path completi**: Vedi `config/preprocessing_config.yaml` per reference
- **Notebook**: `notebooks/PreProcessing.ipynb`
