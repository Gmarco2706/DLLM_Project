# Fase 2: Corruption (Split e Corruzione Dati)

## Overview

La Fase 2 si concentra sulla preparazione dei dati per la successiva fase di imputazione e valutazione. I notebook relativi a questa fase si trovano nella directory `notebooks/Fase2/`. Questa fase è fondamentale per poter validare oggettivamente le tecniche per la gestione dei dati mancanti (missing values).

## Step della Fase 2

### 1. Split del Dataset (`SplitDataset.ipynb`)
- **Obiettivo**: Suddividere il dataset pulito in set di addestramento (Train), validazione (Validation) e test (Test).
- **Motivazione**: Garantire una valutazione rigorosa dei modelli, evitando *data leakage* tra i dati usati per apprendere le tecniche di imputazione/predizione e i dati su cui queste tecniche vengono verificate.
- **Dati di input**: Dataset pre-processato finale (risultante dalla Fase 1).

### 2. Corruzione dei Dati (`DataCorruption.ipynb`)
- **Obiettivo**: Introdurre deliberatamente valori mancanti (missing values) nel dataset originariamente completo seguendo uno specifico "Schema di Corruzione".
- **Motivazione**: Per testare rigorosamente l'efficacia dei modelli di imputazione è necessario disporre della *ground truth*. Corrompendo in modo controllato un dataset integro, si potrà in seguito misurare esattamente quanto le ricostruzioni algoritmiche si avvicinino al valore originale.

---

**Next Step**: Una volta ottenuto il dataset artificialmente corrotto assieme alla sua controparte integra (ground truth), si passa alla **Fase 3: Imputation** in cui i dati mancanti vengono trattati e ricostruiti.
