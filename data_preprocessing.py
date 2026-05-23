import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def load_and_preprocess_diabetes(n_qubits=4, test_size=0.2, random_state=42):
    # 1. Caricamento del dataset da scikit-learn
    diabetes = load_diabetes()
    X = diabetes.data
    y_reg = diabetes.target  # Target continuo (per il Membro 2 - VQR)

    # 2. Creazione Target binario (per il Membro 1 - VQC)
    # Dividiamo i pazienti a metà usando la mediana del target
    median_val = np.median(y_reg)
    y_class = (y_reg > median_val).astype(int)

    # 3. Standardizzazione delle feature
    # Nota: scikit-learn fornisce Diabetes già parzialmente scalato, 
    # ma riapplicare lo standard scaler è best practice prima della PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Riduzione dimensionale con PCA (10 feature -> n_qubits)
    pca = PCA(n_components=n_qubits)
    X_pca = pca.fit_transform(X_scaled)

    # 5. Adattamento per i circuiti quantistici (Quantum Feature Map)
    # Scaliamo i valori tra 0 e 1 (ottimale per rotazioni Ry/Rz o ZZFeatureMap)
    minmax = MinMaxScaler(feature_range=(0, 1))
    X_quantum_ready = minmax.fit_transform(X_pca)

    # 6. Split in Train e Test set per entrambi i modelli
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X_quantum_ready, y_class, y_reg, test_size=test_size, random_state=random_state
    )

    # Stampa la varianza spiegata per inserirla nella relazione
    print(f"Dataset originale: {X.shape[1]} features.")
    print(f"Varianza spiegata da {n_qubits} componenti principali (qubit): {sum(pca.explained_variance_ratio_):.2%}")

    return {
        "classification": (X_train, X_test, y_class_train, y_class_test),
        "regression": (X_train, X_test, y_reg_train, y_reg_test),
        "median_threshold": median_val  # Utile da sapere per l'analisi dati
    }


# Test rapido dello script
if __name__ == "__main__":
    data = load_and_preprocess_diabetes(n_qubits=4)
    print("Shape dati Train (X):", data["classification"][0].shape)
    print("Soglia mediana usata per la classificazione:", data["median_threshold"])