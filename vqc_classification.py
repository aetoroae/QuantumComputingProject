import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, log_loss)
from scipy.optimize import minimize

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
from qiskit.primitives import StatevectorSampler

from data_preprocessing import load_and_preprocess_diabetes

# FUNZIONI DI CREAZIONE CIRCUITI (ENCODING E ANSATZ)
def create_encoding_circuits(n_qubits):
    """Crea i diversi tipi di encoding quantistico (3 configurazioni)"""
    encodings = {}
    encodings['ZFeatureMap'] = ZFeatureMap(feature_dimension=n_qubits, reps=1)
    encodings['ZZFeatureMap'] = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
    encodings['PauliFeatureMap'] = PauliFeatureMap(feature_dimension=n_qubits, paulis=['XX', 'YY', 'ZZ'], reps=1)
    return encodings


def create_ansatz_circuits(n_qubits):
    """Crea i diversi tipi di ansatz variazionali (3 configurazioni)"""
    ansatz_dict = {}
    ansatz_dict['RealAmplitudes'] = RealAmplitudes(n_qubits, entanglement='circular', reps=2)
    ansatz_dict['TwoLocal'] = TwoLocal(n_qubits, 'ry', 'cx', 'reverse_linear', reps=2)
    ansatz_dict['EfficientSU2'] = EfficientSU2(n_qubits, reps=1)
    return ansatz_dict


class QuantumDiabetesClassifier:
    def __init__(self, feature_map, ansatz):
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.circuit = feature_map.compose(ansatz)
        self.circuit.measure_all()
        self.history = []
        self.optimal_params = None

    def circuit_instance(self, data_point, variational_params):
        """Crea un'istanza del circuito con parametri specifici"""
        parameters = {}
        for i, p in enumerate(self.feature_map.ordered_parameters):
            parameters[p] = data_point[i]
        for i, p in enumerate(self.ansatz.ordered_parameters):
            parameters[p] = variational_params[i]
        return self.circuit.assign_parameters(parameters)

    def interpreter(self, bitstring):
        """Interpreta la stringa di bit come classe (0 o 1) tramite Hamming Weight"""
        hamming_weight = sum(int(k) for k in list(bitstring))
        return hamming_weight % 2

    def label_probability(self, results):
        """Calcola le probabilità per ogni classe (0, 1)"""
        shots = sum(results.values())
        probabilities = {0: 0, 1: 0}
        for bitstring, counts in results.items():
            label = self.interpreter(bitstring)
            probabilities[label] += counts / shots
        return probabilities

    def classification_probability(self, data, variational_params):
        """Calcola le probabilità di classificazione per un batch di dati"""
        circuits = [self.circuit_instance(point, variational_params) for point in data]
        sampler = StatevectorSampler()
        results = sampler.run(circuits).result()

        classifications = []
        for i, circuit in enumerate(circuits):
            probs = self.label_probability(results[i].data.meas.get_counts())
            classifications.append(probs)
        return classifications

    def cost_function(self, variational_params, data, labels):
        """Funzione di costo (cross-entropy loss) - Caso binario"""
        classifications = self.classification_probability(data, variational_params)
        y_pred = [[p[0], p[1]] for p in classifications]
        cost = log_loss(y_true=labels, y_pred=y_pred, labels=[0, 1])
        self.history.append(cost)
        return cost

    def train(self, train_data, train_labels, optimizer='COBYLA', max_iter=5000):
        """Addestra il classificatore quantistico"""
        self.history = []
        initial_params = np.random.uniform(0, 2 * np.pi, self.ansatz.num_parameters)

        objective = lambda params: self.cost_function(params, train_data, train_labels)
        start_time = time.time()

        if optimizer == 'COBYLA':
            result = minimize(objective, initial_params, method='COBYLA', options={'maxiter': max_iter})
        elif optimizer == 'L_BFGS_B':
            result = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': max_iter})
        elif optimizer == 'SLSQP':
            result = minimize(objective, initial_params, method='SLSQP', options={'maxiter': max_iter})
        else:
            raise ValueError(f"Optimizer {optimizer} non supportato")

        training_time = time.time() - start_time
        self.optimal_params = result.x
        final_cost = result.fun

        return {
            'optimal_params': self.optimal_params,
            'final_cost': final_cost,
            'training_time': training_time,
            'iterations': len(self.history),
            'success': result.success
        }

    def predict(self, data):
        """Predizione usando i parametri ottimali"""
        if self.optimal_params is None:
            raise ValueError("Modello non addestrato")
        probabilities = self.classification_probability(data, self.optimal_params)
        predictions = [max(p, key=p.get) for p in probabilities]
        return predictions, probabilities

    def evaluate(self, data, labels):
        """Valuta le performance del modello"""
        predictions, _ = self.predict(data)
        accuracy = accuracy_score(labels, predictions)
        return accuracy, predictions


def run_comparative_experiments(train_features, train_labels, test_features, test_labels, n_qubits, target_names):
    """Esegue la Grid Search completa (27 esperimenti) e salva i risultati in una cartella"""

    # Crea la cartella per i risultati se non esiste
    output_dir = "risultati_vqc"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Grafici e score verranno salvati nella cartella: '{output_dir}'")

    print("\n" + "=" * 80)
    print("ESPERIMENTI COMPARATIVI VQC - DIABETES DATASET (GRID SEARCH 3x3x3)")
    print("=" * 80)

    encodings = create_encoding_circuits(n_qubits)
    ansatz_circuits = create_ansatz_circuits(n_qubits)

    # I 3 OTTIMIZZATORI RIPRISTINATI
    optimizers = ['COBYLA', 'L_BFGS_B', 'SLSQP']

    results = []
    experiment_count = 0
    total_experiments = len(encodings) * len(ansatz_circuits) * len(optimizers)

    for enc_name, encoding in encodings.items():
        for ans_name, ansatz in ansatz_circuits.items():
            for optimizer in optimizers:
                experiment_count += 1
                print(f"\n{'=' * 20} ESPERIMENTO {experiment_count}/{total_experiments} {'=' * 20}")
                print(f"Encoding: {enc_name} | Ansatz: {ans_name} | Opt: {optimizer}")

                classifier = QuantumDiabetesClassifier(encoding, ansatz)

                print("Addestramento in corso (potrebbe richiedere tempo)...")
                train_results = classifier.train(
                    train_features, train_labels,
                    optimizer=optimizer,
                    max_iter=200  # Ripristinato a 200 iterazioni per la massima precisione
                )

                train_acc, train_pred = classifier.evaluate(train_features, train_labels)
                test_acc, test_pred = classifier.evaluate(test_features, test_labels)

                test_precision = precision_score(test_labels, test_pred, average='weighted', zero_division=0)
                test_recall = recall_score(test_labels, test_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(test_labels, test_pred, average='weighted', zero_division=0)

                print(f"RISULTATI:")
                print(f" Test Accuracy:  {test_acc:.4f} | F1-Score: {test_f1:.4f}")
                print(
                    f" Final Cost:     {train_results['final_cost']:.4f} | Tempo: {train_results['training_time']:.2f}s")

                experiment_title = f"{enc_name}_{ans_name}_{optimizer}"

                # Plot e Salvataggio Confusion Matrix (In alta definizione per la relazione)
                cm = confusion_matrix(test_labels, test_pred)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
                plt.title(f'Confusion Matrix\n{enc_name} + {ans_name} + {optimizer}')
                plt.ylabel('True')
                plt.xlabel('Predicted')
                plt.savefig(os.path.join(output_dir, f'CM_{experiment_title}.png'), bbox_inches='tight', dpi=300)
                plt.show()
                plt.close()

                # Plot e Salvataggio curva di convergenza della Loss
                if len(classifier.history) > 1:
                    plt.figure(figsize=(6, 4))
                    plt.plot(classifier.history, 'b-', linewidth=2)
                    plt.title(f'Training Convergence\n{enc_name} + {ans_name} + {optimizer}')
                    plt.xlabel('Iteration')
                    plt.ylabel('Cost (Log Loss)')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(output_dir, f'Loss_{experiment_title}.png'), bbox_inches='tight', dpi=300)
                    plt.show()
                    plt.close()

                results.append({
                    'encoding': enc_name, 'ansatz': ans_name, 'optimizer': optimizer,
                    'test_accuracy': test_acc, 'test_precision': test_precision,
                    'test_recall': test_recall, 'test_f1': test_f1,
                    'final_cost': train_results['final_cost'],
                    'training_time': train_results['training_time'],
                    'iterations': train_results['iterations'],
                    'success': train_results['success']
                })
    return results


def analyze_results(results, n_components):
    """Sintesi finale delle metriche ed esportazione automatica in un report CSV"""
    df_results = pd.DataFrame(results)

    # Esportazione della tabella riassuntiva in CSV leggibile da Excel
    output_dir = "risultati_vqc"
    csv_path = os.path.join(output_dir, "metriche_esperimenti_completi.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[INFO] Tabella finale dei punteggi esportata con successo in: {csv_path}")

    print("\n" + "=" * 80)
    print("ANALISI AVANZATA DEI RISULTATI")
    print("=" * 80)

    print(f"\nAccuracy media quantistica (test): {df_results['test_accuracy'].mean():.4f}")
    print(f"Tempo medio di training: {df_results['training_time'].mean():.2f}s")

    print("\nTOP CONFIGURAZIONI (Test Accuracy)")
    print("-" * 80)
    top_configs = df_results.nlargest(10, 'test_accuracy')
    print(f"{'Rank':<4} {'Encoding':<15} {'Ansatz':<15} {'Optimizer':<10} {'Test Acc':<8} {'Time(s)':<7}")
    for rank, (idx, row) in enumerate(top_configs.iterrows(), 1):
        print(f"{rank:<4} {row['encoding']:<15} {row['ansatz']:<15} {row['optimizer']:<10} "
              f"{row['test_accuracy']:<8.4f} {row['training_time']:<7.1f}")


if __name__ == "__main__":
    # 1. Caricamento dati dal tuo file di preprocessing esterno
    n_qubits = 4
    data = load_and_preprocess_diabetes(n_qubits=n_qubits)
    X_train, X_test, y_class_train, y_class_test = data["classification"]

    target_names = ['Bassa Progr.', 'Alta Progr.']  # Classificazione binaria per il Diabetes binarizzato

    # 2. Avvio degli esperimenti quantistici controllati (27 totali)
    quantum_results = run_comparative_experiments(
        X_train, y_class_train,
        X_test, y_class_test,
        n_qubits, target_names
    )

    # 3. Analisi e generazione dei report finali
    analyze_results(quantum_results, n_qubits)