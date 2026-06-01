import numpy as np

from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# 1. AUTENTICAZIONE
IBM_TOKEN = "segreta"

print("[1] Autenticazione ai server IBM in corso...")
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)

# Trova in automatico il computer reale con la coda più corta
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=4)
print(f" -> Connesso con successo! Il computer selezionato è: {backend.name}")

# Import dei dati
from data_preprocessing import load_and_preprocess_diabetes
data = load_and_preprocess_diabetes(n_qubits=4)
X_train, X_test, y_class_train, y_class_test = data["classification"]

# 2. ADDESTRAMENTO LOCALE
print("\n[2] Addestramento locale del modello in corso (Mac M4)...")
feature_map = ZFeatureMap(feature_dimension=4, reps=1)
ansatz = RealAmplitudes(4, entanglement='circular', reps=2)
circuit = feature_map.compose(ansatz)
circuit.measure_all()

# Usiamo il simulatore locale per il training
from qiskit.primitives import StatevectorSampler
local_sampler = StatevectorSampler()

def interpreter(bitstring):
    return sum(int(k) for k in list(bitstring)) % 2  

def get_cost(params, x_data, y_data):
    # Creazione circuiti
    bound_circuits = []
    for point in x_data:
        parameters = {}
        for i, p in enumerate(feature_map.ordered_parameters): parameters[p] = point[i]
        for i, p in enumerate(ansatz.ordered_parameters): parameters[p] = params[i]
        bound_circuits.append(circuit.assign_parameters(parameters))
    
    # Esecuzione locale
    results = local_sampler.run(bound_circuits).result()
    cost = 0
    for i, res in enumerate(results):
        counts = res.data.meas.get_counts()
        shots = sum(counts.values())
        prob_1 = sum(counts[k] for k in counts if interpreter(k) == 1) / shots
        y_true = y_data[i]
        # Cross-entropy semplificata
        prob_1 = max(min(prob_1, 0.999), 0.001)
        cost -= (y_true * np.log(prob_1) + (1 - y_true) * np.log(1 - prob_1))
    return cost / len(x_data)

# Training
init_params = np.random.uniform(0, 2*np.pi, ansatz.num_parameters)
res = minimize(lambda p: get_cost(p, X_train[:30], y_class_train[:30]), init_params, method='COBYLA', options={'maxiter': 30})
optimal_params = res.x
print(" -> Addestramento completato. Pesi ottimali congelati.")

# 3. PREPARAZIONE E INVIO AL VERO COMPUTER IBM
# Prendiamo SOLO 5 PAZIENTI di test
n_samples = 5
X_poc = X_test[:n_samples]
y_poc = y_class_test[:n_samples]

print("\n[3] Preparazione circuiti per l'hardware IBM...")
test_circuits = []
for point in X_poc:
    parameters = {}
    for i, p in enumerate(feature_map.ordered_parameters): parameters[p] = point[i]
    for i, p in enumerate(ansatz.ordered_parameters): parameters[p] = optimal_params[i]
    test_circuits.append(circuit.assign_parameters(parameters))

# Transpilazione
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
isa_circuits = pm.run(test_circuits)

print(f"\n[4] INVIO DEL JOB AL COMPUTER QUANTISTICO '{backend.name}'...")
sampler = Sampler(mode=backend)
job = sampler.run(isa_circuits)

print("="*60)
print(f"JOB INVIATO CON SUCCESSO! ID: {job.job_id()}")
print("="*60)
print("Il tuo Mac è ora in attesa che IBM completi l'esecuzione fisica.")
print("ATTENZIONE: Potrebbero volerci minuti o ore a seconda della coda globale.")
print("Non chiudere il programma se vuoi vedere i risultati qui sotto.\n")

try:
    # Questo comando mette Python in pausa finché IBM non ha finito
    result = job.result()
    print("\n ESECUZIONE FISICA COMPLETATA!")
    
    correct_predictions = 0
    for i in range(n_samples):
        counts = result[i].data.meas.get_counts()
        shots = sum(counts.values())
        probs = {0: 0, 1: 0}
        for bitstring, count in counts.items():
            probs[interpreter(bitstring)] += count / shots
            
        prediction = max(probs, key=probs.get)
        actual = y_poc[i]
        is_correct = "CORRETTO" if prediction == actual else "ERRATO"
        if prediction == actual: correct_predictions += 1
        
        print(f"Paziente {i+1} -> Reale: {actual} | Predizione IBM: {prediction} [{is_correct}]")
        
    print(f"\nAccuratezza sul Proof of Concept IBM: {(correct_predictions/n_samples)*100:.1f}%")

except Exception as e:
    print(f"\nC'è stato un problema durante l'attesa o il recupero: {e}")