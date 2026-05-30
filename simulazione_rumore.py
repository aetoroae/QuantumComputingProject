# =============================================================================
# ESPERIMENTO FINALE: TEST DI RESILIENZA AL RUMORE QUANTISTICO (NISQ)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from scipy.optimize import minimize

# Qiskit base imports
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Aer (per la simulazione del rumore hardware)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Il tuo file di preprocessing
from data_preprocessing import load_and_preprocess_diabetes

# =============================================================================
# 1. CLASSE CLASSIFICATORE (Semplificata per il modello vincente)
# =============================================================================
class QuantumDiabetesClassifier:
    def __init__(self, n_qubits):
        self.feature_map = ZFeatureMap(feature_dimension=n_qubits, reps=1)
        self.ansatz = RealAmplitudes(n_qubits, entanglement='circular', reps=2)
        self.circuit = self.feature_map.compose(self.ansatz)
        self.circuit.measure_all()
        self.history = []
        self.optimal_params = None
        
    def circuit_instance(self, data_point, variational_params):
        parameters = {}
        for i, p in enumerate(self.feature_map.ordered_parameters):
            parameters[p] = data_point[i]
        for i, p in enumerate(self.ansatz.ordered_parameters):
            parameters[p] = variational_params[i]
        return self.circuit.assign_parameters(parameters)

    def interpreter(self, bitstring):
        return sum(int(k) for k in list(bitstring)) % 2  

    def label_probability(self, results):
        shots = sum(results.values())
        probabilities = {0: 0, 1: 0}
        for bitstring, counts in results.items():
            label = self.interpreter(bitstring)
            probabilities[label] += counts / shots
        return probabilities

    def cost_function(self, variational_params, data, labels):
        circuits = [self.circuit_instance(point, variational_params) for point in data]
        sampler = StatevectorSampler() 
        results = sampler.run(circuits).result()
        
        classifications = []
        for i, _ in enumerate(circuits):
            probs = self.label_probability(results[i].data.meas.get_counts()) 
            classifications.append(probs)
            
        y_pred = [[p[0], p[1]] for p in classifications] 
        cost = log_loss(y_true=labels, y_pred=y_pred, labels=[0, 1]) 
        self.history.append(cost)
        return cost

    def train_ideal(self, train_data, train_labels):
        print("\n[1] Addestramento Ideale in corso (COBYLA)...")
        initial_params = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters)
        objective = lambda params: self.cost_function(params, train_data, train_labels)
        
        # Facciamo 100 iterazioni per un addestramento veloce e mirato
        result = minimize(objective, initial_params, method='COBYLA', options={'maxiter': 100})
        self.optimal_params = result.x
        return result.fun

    def evaluate_ideal(self, test_data, test_labels):
        circuits = [self.circuit_instance(point, self.optimal_params) for point in test_data]
        sampler = StatevectorSampler() 
        results = sampler.run(circuits).result()
        
        predictions = []
        for i, _ in enumerate(circuits):
            probs = self.label_probability(results[i].data.meas.get_counts()) 
            predictions.append(max(probs, key=probs.get))
            
        return accuracy_score(test_labels, predictions), predictions

    def evaluate_noisy(self, test_data, test_labels, noise_level=0.15):
        print(f"\n[2] Valutazione su Hardware Rumoroso (Errore CX: {noise_level*100}%)...")
        
        # Creazione del modello di rumore quantistico
        noise_model = NoiseModel()
        # Errore dell'1% sulle porte a singolo qubit (es. rotazioni)
        error_1q = depolarizing_error(noise_level / 5, 1)
        # Errore del 5% (variabile) sulle porte di entanglement a 2 qubit (CNOT)
        error_2q = depolarizing_error(noise_level, 2)
        
        noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'rz', 'h', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        
        # Setup del simulatore Aer con rumore
        sim = AerSimulator(noise_model=noise_model)
        pm = generate_preset_pass_manager(optimization_level=1, backend=sim)
        
        predictions = []
        # Valutazione punto per punto
        for point in test_data:
            bound_circ = self.circuit_instance(point, self.optimal_params)
            transpiled_circ = pm.run(bound_circ)
            result = sim.run(transpiled_circ, shots=1024).result()
            
            probs = self.label_probability(result.get_counts())
            predictions.append(max(probs, key=probs.get))
            
        return accuracy_score(test_labels, predictions), predictions

# =============================================================================
# 2. ESECUZIONE E PLOT
# =============================================================================
if __name__ == "__main__":
    
    # Crea cartella output
    output_dir = "classificazione_vqc"
    os.makedirs(output_dir, exist_ok=True)
    
    n_qubits = 4
    data = load_and_preprocess_diabetes(n_qubits=n_qubits)
    X_train, X_test, y_class_train, y_class_test = data["classification"]
    target_names = ['Bassa Progr.', 'Alta Progr.']
    
    classifier = QuantumDiabetesClassifier(n_qubits)
    
    # Addestramento
    classifier.train_ideal(X_train, y_class_train)
    
    # Test Ideale vs Reale
    acc_ideal, pred_ideal = classifier.evaluate_ideal(X_test, y_class_test)
    acc_noisy, pred_noisy = classifier.evaluate_noisy(X_test, y_class_test, noise_level=0.05)
    
    print("\n" + "="*50)
    print(f" RISULTATI FINALI (ZFeatureMap + RealAmplitudes)")
    print("="*50)
    print(f" Accuracy Ideale (Simulatore Perfetto): {acc_ideal*100:.2f}%")
    print(f" Accuracy Reale  (Rumore IBM al 15%):    {acc_noisy*100:.2f}%")
    
    # PLOT: Grafico a Barre Comparativo
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Ideale (Senza Rumore)', 'Rumoroso (Errore 15%)'], [acc_ideal, acc_noisy], color=['#1f77b4', '#d62728'])
    plt.ylabel('Test Accuracy')
    plt.title('Resilienza al Rumore Hardware (NISQ)\nModello: ZFeatureMap + RealAmplitudes')
    plt.ylim(0, 1.0)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random Guess (50%)')
    plt.legend()
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.savefig(os.path.join(output_dir, 'Confronto_Rumore.png'), bbox_inches='tight', dpi=300)
    plt.show()