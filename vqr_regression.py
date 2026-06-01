import numpy as np
import time

# --- Configurazione per forzare la finestra esterna dei grafici ---
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.algorithms.regressors import VQR

from data_preprocessing import load_and_preprocess_diabetes

num_qubits = 4
data_dict = load_and_preprocess_diabetes(n_qubits=num_qubits)

X_train, X_test, y_train, y_test = data_dict["regression"]

#  SCALING DEL TARGET
y_scaler = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()


print("\n--- 1. Addestramento Modelli Classici ---")
# Regressione Lineare
lin_reg = LinearRegression().fit(X_train, y_train)
pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, pred_lin)

# Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, pred_rf)

print(f"Regressione Lineare -> MSE: {mse_lin:.2f}")
print(f"Random Forest       -> MSE: {mse_rf:.2f}")


estimator = StatevectorEstimator()
feature_map = zz_feature_map(feature_dimension=num_qubits, reps=2, entanglement='linear')


class TrackedCOBYLA(COBYLA):
    def __init__(self, loss_history_list, **kwargs):
        super().__init__(**kwargs)
        self.loss_history_list = loss_history_list

    def minimize(self, fun, x0, jac=None, bounds=None, *args, **kwargs):
        def wrapped_fun(x):
            val = fun(x)
            val_scalar = val.item() if hasattr(val, 'item') else val
            self.loss_history_list.append(val_scalar)
            return val

        return super().minimize(wrapped_fun, x0, jac=jac, bounds=bounds, *args, **kwargs)



print("\n--- 2. Addestramento VQR Varianti ---")
vqr_results = {}

# Configurazione: indichiamo solo i nomi e il tipo di ottimizzatore da usare
runs = [
    ("COBYLA_reps1", 1, "COBYLA"),
    ("COBYLA_reps3", 3, "COBYLA"),
    ("SPSA_reps3", 3, "SPSA")
]

for name, depth, opt_type in runs:
    print(f"\n> Addestramento {name} in corso...")
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=depth)

    loss_history = []

    # Assegnazione dinamica dell'ottimizzatore
    if opt_type == "COBYLA":
        optimizer = TrackedCOBYLA(loss_history_list=loss_history, maxiter=80)
        callback = None
    else:
        optimizer = SPSA(maxiter=80)


        def callback(*args):
            if len(args) == 5:
                val = args[2]
                val_scalar = val.item() if hasattr(val, 'item') else val
                loss_history.append(val_scalar)

    vqr = VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
        callback=callback
    )

    start = time.time()
    vqr.fit(X_train, y_train_scaled)
    print(f"  Completato in {time.time() - start:.2f} s")

    # Predizione quantistica
    pred_scaled = vqr.predict(X_test)

    # Riportiamo il risultato alla scala medica reale del diabete
    pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    mse_vqr = mean_squared_error(y_test, pred_real)
    r2_vqr = r2_score(y_test, pred_real)

    vqr_results[name] = {
        'mse': mse_vqr,
        'r2': r2_vqr,
        'loss': loss_history,
        'predictions': pred_real
    }
    print(f"  Risultato -> MSE: {mse_vqr:.2f} | R2: {r2_vqr:.2f}")

# VISUALIZZAZIONE RISULTATI
plt.figure(figsize=(18, 5))

# Grafico 1: Curve di Loss a confronto
plt.subplot(1, 3, 1)
plt.plot(vqr_results['COBYLA_reps1']['loss'], label='COBYLA (reps=1)', color='lightblue')
plt.plot(vqr_results['COBYLA_reps3']['loss'], label='COBYLA (reps=3)', color='blue')
plt.plot(vqr_results['SPSA_reps3']['loss'], label='SPSA (reps=3)', color='orange')
plt.title("Convergenza: Ottimizzatori e Profondità")
plt.xlabel("Iterazioni")
plt.ylabel("Cost Function (MSE scalato)")
plt.legend()
plt.grid(True)

# Grafico 2: Confronto MSE Finale
plt.subplot(1, 3, 2)
modelli = ['Lin. Reg', 'Rand. Forest', 'VQR (C-r1)', 'VQR (C-r3)', 'VQR (S-r3)']
mse_values = [mse_lin, mse_rf, vqr_results['COBYLA_reps1']['mse'], vqr_results['COBYLA_reps3']['mse'],
              vqr_results['SPSA_reps3']['mse']]
colors = ['gray', 'gray', 'lightblue', 'blue', 'orange']
plt.bar(modelli, mse_values, color=colors)
plt.title("Confronto MSE sul Test Set (Più basso è meglio)")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=25)
for i, v in enumerate(mse_values):
    plt.text(i, v + 50, f"{v:.0f}", ha='center')

# Grafico 3: Dispersione - Reale vs Predetto
plt.subplot(1, 3, 3)
plt.scatter(y_test, pred_rf, alpha=0.4, label='Random Forest', color='gray')
plt.scatter(y_test, vqr_results['COBYLA_reps3']['predictions'], alpha=0.6, label='VQR COBYLA (r=3)', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Predizione Ideale')
plt.title("Diabete Reale vs Predetto")
plt.xlabel("Target Reale")
plt.ylabel("Target Predetto")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()