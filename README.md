# Alice & Bob Qiskit provider

This project contains a provider that allows access to
[Alice & Bob](https://alice-bob.com/) QPUs and emulators using
the Qiskit framework.

Full documentation
[is available here](https://felis.alice-bob.com/docs/)
and sample notebooks using the provider
[are available here](https://github.com/Alice-Bob-SW/felis/tree/main/samples).

## Installation

You can install the provider using `pip`:

```bash
pip install qiskit-alice-bob-provider
```

`pip` will handle installing all the python dependencies automatically and you
will always install the latest (and well-tested) version.

> [!WARNING]
> Transpilation of gates CRY, RCCX and RCCCX does not work on macOS currently.

## Remote execution on Alice & Bob QPUs: use your API key

To obtain an API key, get a Felis Cloud subscription on the [Google Cloud Marketplace](https://console.cloud.google.com/marketplace/product/cloud-prod-0/felis-cloud) or [contact Alice & Bob](https://alice-bob.com/contact/).

You can initialize the Alice & Bob remote provider using your API key
locally with:

```python
from qiskit_alice_bob_provider import AliceBobRemoteProvider
ab = AliceBobRemoteProvider('MY_API_KEY')
```

Where `MY_API_KEY` is your API key to the Alice & Bob API.

```python
print(ab.backends())
backend = ab.get_backend('EMU:1Q:LESCANNE_2020')
```

The backend can then be used like a regular Qiskit backend:

```python
from qiskit import QuantumCircuit

c = QuantumCircuit(1, 2)
c.initialize('+', 0)
c.measure_x(0, 0)
c.measure(0, 1)
job = backend.run(c)
res = job.result()
print(res.get_counts())
```

## Local emulation of cat qubit processors

This project contains multiple emulators of multi cat qubit processors.

```python
from qiskit_alice_bob_provider import AliceBobLocalProvider
from qiskit import QuantumCircuit, transpile

provider = AliceBobLocalProvider()
print(provider.backends())
# EMU:6Q:PHYSICAL_CATS, EMU:40Q:PHYSICAL_CATS, EMU:1Q:LESCANNE_2020
```

The `EMU:nQ:PHYSICAL_CATS` backends are theoretical models of quantum processors made
up of physical cat qubits.
They can be used to study the properties of error correction codes implemented
with physical cat qubits, for different hardware performance levels
(see the parameters of class `PhysicalCatProcessor`).

The `EMU:1Q:LESCANNE_2020` backend is an interpolated model simulating the processor
used in the [seminal paper](https://arxiv.org/pdf/1907.11729.pdf) by Raphaël
Lescanne in 2020.
This interpolated model is configured to act as a digital twin of the cat qubit
used in this paper.
It does not represent the current performance of Alice & Bob's cat qubits.

The example below schedules and simulates a Bell state preparation circuit on
a `EMU:6Q:PHYSICAL_CATS` processor, for different values of parameters
`average_nb_photons` and `kappa_2`.

```python
from qiskit_alice_bob_provider import AliceBobLocalProvider
from qiskit import QuantumCircuit, transpile

provider = AliceBobLocalProvider()

circ = QuantumCircuit(2, 2)
circ.initialize('0+')
circ.cx(0, 1)
circ.measure(0, 0)
circ.measure(1, 1)

# Default 6-qubit QPU with the ratio of memory dissipation rates set to
# k1/k2=1e-5 and cat size, average_nb_photons, set to 16.
backend = provider.get_backend('EMU:6Q:PHYSICAL_CATS')

print(transpile(circ, backend).draw())
# *Displays a timed and scheduled circuit*

print(backend.run(circ, shots=100000).result().get_counts())
# {'11': 49823, '00': 50177}

# Changing the cat size from 16 (default) to 4 and k1/k2 to 1e-2.
backend = provider.get_backend(
    'EMU:6Q:PHYSICAL_CATS', average_nb_photons=4, kappa_2=1e4
)
print(backend.run(circ, shots=100000).result().get_counts())
# {'01': 557, '11': 49422, '10': 596, '00': 49425}
```

## Setting Up Development Environment (for contributors only)

To ensure code consistency and enforce commit message conventions, we use `pre-commit` (Python-based) and `commitlint` (Node.js-based). Follow these steps to set up your development environment.

### Prerequisites

You need the following installed on your system:

- **Python 3.12**
- **Node.js** (latest LTS version recommended, required for `commitlint`)
- **pnpm** (used to install `commitlint` dependencies)

### Installation Steps

1. **Install Python dependencies**  
   Run the following command to set up the Python environment and install dependencies:

   ```bash
   make install
   ```

   This will:

   - Create a Python **virtual environment**
   - Install all required dependencies, including `pre-commit`

2. **Install commitlint dependencies**  
   Run the following command to install the necessary Node.js packages:

   ```bash
   pnpm install
   ```

3. **Install Git hooks**  
   Run the following command to install all required Git hooks:
   ```bash
   make precommit-hooks
   ```

Happy coding! 🚀
