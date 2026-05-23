# 🧬 Quantum Machine Learning for Diabetes Analysis (VQR & VQC)

[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-blue.svg?logo=qiskit)](https://qiskit.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Questo repository contiene un progetto completo di **Quantum Machine Learning (QML)** applicato al dataset *Diabetes* di scikit-learn. Il progetto esplora le architetture dei circuiti quantistici variazionali affrontando il problema sia dal punto di vista della **Regressione Continua (VQR)** sia della **Classificazione Binaria (VQC)**.

Progetto di gruppo realizzato per l'esame di **Quantum Computing**.

---

## 📋 Indice
1. [Descrizione del Progetto e Preprocessing Condiviso](#1-descrizione-del-progetto-e-preprocessing-condiviso)
2. [Modulo A: Variational Quantum Regression (VQR)](#2-modulo-a-variational-quantum-regression-vqr)
3. [Modulo B: Variational Quantum Classification (VQC)](#3-modulo-b-variational-quantum-classification-vqc)
4. [Prerequisiti e Installazione](#4-prerequisiti-e-installazione)
5. [Esecuzione del Codice](#5-esecuzione-del-codice)

---

## 1. Descrizione del Progetto e Preprocessing Condiviso
Il progetto utilizza un approccio ibrido quantistico-classico per analizzare dati clinici. 
Il file `data_preprocessing.py` costituisce il modulo condiviso tra tutti i membri del team e si occupa di preparare i dati per i computer quantistici (limitati attualmente a pochi qubit).

**Pipeline di Preprocessing:**
* **Standardizzazione** delle 10 feature originali.
* **Riduzione Dimensionale (PCA)** per comprimere le informazioni in $n$ componenti principali (es. 4 qubit).
* **Quantum Scaling** (MinMaxScaler [0, 1]) per mappare i valori in angoli fisicamente validi per le rotazioni dei qubit.
* **Target Split:** Estrazione del target continuo per la regressione e creazione di un target binario (soglia basata sulla mediana) per la classificazione.

---

## 2. Modulo A: Variational Quantum Regression (VQR)
*(Sviluppato da: [Tuo Nome/Username])*

Questa sezione si occupa di prevedere l'avanzamento quantitativo della malattia analizzando il target continuo tramite la classe `VQR` di Qiskit.

### 🏗️ Architettura del Modello
* **Data Encoding:** `zz_feature_map` (4 Qubit, `reps=2`). Agisce come kernel quantistico non-lineare generando entanglement tra le feature cliniche.
* **Ansatz:** `real_amplitudes` (4 Qubit). Circuito variazionale *Hardware-Efficient* per addestrare i pesi minimizzando numeri complessi e Barren Plateaus.
* **Misurazione:** `StatevectorEstimator()`. Calcola il Valore Atteso dell'operatore Z (riscalato nel range medico tramite `inverse_transform`).

### ⚙️ Analisi degli Ottimizzatori e Iperparametri
Il modulo implementa un sistema di Benchmark (attraverso un OOP Wrapper custom) per confrontare:
1.  **L'impatto della profondità:** Confronto dell'espressività tra circuiti *Shallow* (`reps=1`) e *Deep* (`reps=3`).
2.  **SPSA vs COBYLA:** * **COBYLA:** Derivative-free, dimostra una discesa estremamente rapida ed efficiente nei simulatori software ideali.
    * **SPSA:** Gradiente stocastico perturbato. Richiede solo due valutazioni per step e la sua natura stocastica lo rende la scelta d'elezione per assorbire il rumore in futuri hardware quantistici reali (NISQ).
3.  **Benchmark Classico:** Confronto diretto delle metriche (MSE, $R^2$) con *Linear Regression* e *Random Forest*.

---

## 3. Modulo B: Variational Quantum Classification (VQC)
*(Sviluppato da: [Nome del tuo collega / Placeholder])*

> **[ATTENZIONE TEAM: Inserire qui la documentazione del VQC]**
> * *Descrivere la struttura del VQC (es. Sampler invece di Estimator).*
> * *Descrivere la funzione di costo (es. Cross-Entropy).*
> * *Descrivere l'Ansatz utilizzato o le differenze rispetto alla regressione.*
> * *Inserire i risultati di classificazione (Accuracy, F1-Score).*

---

## 4. Prerequisiti e Installazione
Assicurati di avere Python 3.10 o superiore. Crea un ambiente virtuale e installa le dipendenze richieste:

```bash
# Creazione ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installazione librerie base
pip install numpy matplotlib scikit-learn

# Installazione ecosistema IBM Qiskit (versione 1.x o superiore)
pip install qiskit qiskit-machine-learning qiskit-algorithms
