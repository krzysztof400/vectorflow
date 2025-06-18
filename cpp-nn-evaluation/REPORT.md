# Raport z Trenowania
## Wyniki
Activation Function: ReLU, Learning Rate: 0.001, Accuracy: 100%

Activation Function: Sigmoid, Learning Rate: 0.001, Accuracy: 59%

Activation Function: ReLU (L1 Normalized), Learning Rate: 0.001, Accuracy: 91%

Activation Function: Sigmoid (L1 Normalized), Learning Rate: 0.001, Accuracy: 73%

Activation Function: ReLU (L2 Normalized), Learning Rate: 0.001, Accuracy: 91%

Activation Function: Sigmoid (L2 Normalized), Learning Rate: 0.001, Accuracy: 73%

Activation Function: ReLU (Learning Rate: 0.010000), Learning Rate: 0.01, Accuracy: 99%

Activation Function: Sigmoid (Learning Rate: 0.010000), Learning Rate: 0.01, Accuracy: 98%

Activation Function: ReLU (Learning Rate: 0.001000), Learning Rate: 0.001, Accuracy: 98%

Activation Function: Sigmoid (Learning Rate: 0.001000), Learning Rate: 0.001, Accuracy: 63%

Activation Function: ReLU (Learning Rate: 0.000100), Learning Rate: 0.0001, Accuracy: 93%

Activation Function: Sigmoid (Learning Rate: 0.000100), Learning Rate: 0.0001, Accuracy: 39%

## Obserwacje
- ReLU konsekwentnie przewyższa Sigmoid pod względem dokładności we wszystkich konfiguracjach.
- Wyższe współczynniki uczenia (np. 0.01) zazwyczaj prowadzą do lepszej dokładności dla obu funkcji aktywacji.
- Normalizacja poprawia wydajność Sigmoid, ale nie ma znaczącego wpływu na ReLU.
- Sigmoid ma trudności z niższymi współczynnikami uczenia, co skutkuje niższą dokładnością i wyższą stratą.

