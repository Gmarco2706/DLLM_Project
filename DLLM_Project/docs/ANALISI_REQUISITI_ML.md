# HELOC Dataset Analysis ML

## Overview

Questo documento descrive il processo di analisi delle feature del dataset HELOC (nella sua versione preparata per il Machine Learning, `heloc_dataset_cleanedML.csv`), con l'obiettivo di studiarne le correlazioni reciproche e misurare l'importanza predittiva rispetto alla variabile target `RiskPerformance`.

## Pipeline di Analisi

Il pipeline di analisi è strutturato in step progressivi per studiare tutte le interazioni possibili tra gruppi di variabili aventi nature differenti:

1. **Dataset di input** → `data/processed/heloc_dataset_cleanedML.csv`
2. **Identificazione e separazione dei tipi** → Distinzione automatica tra feature Numeriche e Categoriche. Le colonne binarie (solo 2 valori unici) vengono esplicitamente trattate come categoriche.
3. **Calcolo delle Associazioni** → Elaborazione unificata delle metriche di dipendenza tra vari tipi di feature tramite la libreria `dython`.
4. **Analisi Feature-Target** → Estrazione ed esportazione delle correlazioni specifiche rispetto alla variabile target.

## Associazioni e Metriche

A causa della natura mista del dataset (presenza sia di feature numeriche che categoriche), sono state applicate metodologie statistiche differenziate per il calcolo delle correlazioni:

### 1. Correlazione di Pearson
- **Applicazione**: Feature Numerica vs Feature Numerica
- **Descrizione**: Misura la correlazione lineare tra due variabili continue presenti nel dataset.
- **Output generati**: `heloc_ML_pearson_matrix.csv`, `heloc_ML_pearson_heatmap.png`

### 2. V di Cramér
- **Applicazione**: Feature Categorica vs Feature Categorica
- **Descrizione**: Valuta la forza di associazione tra due variabili nominali. Valori più vicini a 1 indicano una forte associazione.
- **Output generati**: `heloc_ML_cramersv_matrix.csv`, `heloc_ML_cramersv_heatmap.png`

### 3. Rapporto di Correlazione η (Eta)
- **Applicazione**: Feature Categorica vs Feature Numerica
- **Descrizione**: Misura la dispersione della variabile numerica categorizzata rispetto ai gruppi definiti dalla variabile categorica.
- **Output generati**: `heloc_ML_eta_matrix.csv`, `heloc_ML_eta_heatmap.png`

## Estrazione Associazione verso il Target

L'analisi prevede un modulo finale focalizzato sull'ordinamento di importanza (Ranking) delle feature rispetto alla variabile di output (`RiskPerformance`). In base alla natura della metrica utilizzata, è stata generata una classifica strutturata per individuare le feature più promettenti ai fini dell'addestramento.

- **File output**: `data/processed/analysisML/heloc_ML_feature_target_associazioni.csv`
- Vengono esposti i valori assoluti delle metriche (es: Eta, V di Cramér, ecc.) e ordinati in senso decrescente per un'immediata consultazione del loro potere predittivo.

## Implementation Notes

- **Framework**: Python 3.x con librerie `pandas`, `numpy`, `dython`, `seaborn`, `matplotlib`
- **Output Directory**: Tutti i grafici e CSV dell'analisi si trovano in `data/processed/analysisML/`
- **Notebook**: `notebooks/AnalisiRequisitiML.ipynb`
