import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

def load_and_preprocess_diabetes(n_qubits=4, test_size=0.2, random_state=42):
    #Caricamento del dataset da scikit-learn
    diabetes = load_diabetes()
    X = diabetes.data
    y_reg = diabetes.target

    #Creazione Target binario
    #Dividiamo i pazienti a metà usando la mediana del target
    median_val = np.median(y_reg)
    y_class = (y_reg > median_val).astype(int)

    #Standardizzazione delle feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Riduzione dimensionale con PCA (10 feature -> n_qubits)
    pca = PCA(n_components=n_qubits)
    X_pca = pca.fit_transform(X_scaled)

    # Adattamento per i circuiti quantistici (Quantum Feature Map)
    # Scaliamo i valori tra 0 e 1 (ottimale per rotazioni Ry/Rz o ZZFeatureMap)
    minmax = MinMaxScaler(feature_range=(0, 1))
    X_quantum_ready = minmax.fit_transform(X_pca)

    #Split in Train e Test set per entrambi i modelli
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X_quantum_ready, y_class, y_reg, test_size=test_size, random_state=random_state
    )

    #Stampa la varianza spiegata per inserirla nella relazione
    print(f"Dataset originale: {X.shape[1]} features.")
    print(f"Varianza spiegata da {n_qubits} componenti principali (qubit): {sum(pca.explained_variance_ratio_):.2%}")

    return {
        "classification": (X_train, X_test, y_class_train, y_class_test),
        "regression": (X_train, X_test, y_reg_train, y_reg_test),
        "median_threshold": median_val  # Utile da sapere per l'analisi dati
    }


def plot_pca_distribution(X_train, y_train):
    plt.figure(figsize=(8, 6))

    #Plottiamo le prime due componenti (colonna 0 e colonna 1)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1],
                          c=y_train, cmap='coolwarm', alpha=0.8, edgecolors='k')

    plt.title('Distribuzione delle Classi (PCA - Prime 2 Componenti)', fontsize=14)
    plt.xlabel('Componente Principale 1', fontsize=12)
    plt.ylabel('Componente Principale 2', fontsize=12)

    #Aggiungiamo la legenda
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_ticklabels(['Classe 0 (Bassa Progr.)', 'Classe 1 (Alta Progr.)'])

    #Salviamo l'immagine per LaTeX
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('pca_distribution.pdf', format='pdf', bbox_inches='tight')

    print("\nGrafico generato e salvato come 'pca_distribution.pdf'")
    plt.show()


#Test rapido dello script
if __name__ == "__main__":
    data = load_and_preprocess_diabetes(n_qubits=4)
    print("Shape dati Train (X):", data["classification"][0].shape)
    print("Soglia mediana usata per la classificazione:", data["median_threshold"])

    #Estraiamo i dati di training per la classificazione
    X_train, X_test, y_class_train, y_class_test = data["classification"]

    #Stampiamo le dimensioni
    print("\n--- DIMENSIONI DEI TENSORI ---")
    print(f"X_train (Feature): {X_train.shape} -> 353 righe, 4 colonne (qubit)")
    print(f"y_train (Target):  {y_class_train.shape}")

    #Creiamo un DataFrame Pandas per stamparlo in formato tabella
    #Diamo un nome alle 4 colonne (che diventeranno i nostri 4 qubit)
    col_names = [f"PCA_Component_{i + 1}" for i in range(4)]

    df_visual = pd.DataFrame(X_train, columns=col_names)

    #Aggiungiamo la colonna del target per vedere a quale classe appartengono
    df_visual['Target_Classe (0=Bassa, 1=Alta)'] = y_class_train

    print("\n--- PRIME 5 RIGHE DEL DATASET (Pronto per il circuito quantistico) ---")
    print(df_visual.head())

    plot_pca_distribution(X_train,y_class_train)

    #Disabilita il limite delle colonne e allarga la "finestra" virtuale di stampa
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n--- PRIME 5 RIGHE DEL DATASET ---")
    print(df_visual.head())