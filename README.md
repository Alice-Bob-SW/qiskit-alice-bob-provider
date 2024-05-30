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

`pip` will install the latest release with all its dependencies automatically,
including Qiskit.

We recommend creating a new environment before installing the provider, as it
currently **NOT** compatible with Qiskit 1.0.

## Remote execution on Alice & Bob QPUs with a Felis Cloud subscription

When subscribing to Felis Cloud on the
[Google Cloud Marketplace](https://console.cloud.google.com/marketplace/product/cloud-prod-0/felis-cloud),
you get an API key letting you access Alice & Bob QPUs with this provider.

You can initialize the Alice & Bob remote provider using your API key:

```python
from qiskit_alice_bob_provider import AliceBobRemoteProvider
ab = AliceBobRemoteProvider('MY_API_KEY')
```

Where `MY_API_KEY` is the API key of your Felis Cloud subscription.

You may then instantiate the Boson 4 QPU backend:

```python
print(ab.backends())
backend = ab.get_backend('QPU:1Q:BOSON_4A')
```

And use it like a regular Qiskit backend:

```python
from qiskit import QuantumCircuit, execute

c = QuantumCircuit(1, 2)
c.initialize('+', 0)
c.measure_x(0, 0)
c.measure(0, 1)
job = execute(c, backend)
res = job.result()
print(res.get_counts())
```

## Local emulation of cat qubit processors

This project contains multiple emulators of multi cat qubit processors.

```python
from qiskit_alice_bob_provider import AliceBobLocalProvider
from qiskit import QuantumCircuit, execute, transpile

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
from qiskit import QuantumCircuit, execute, transpile

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

print(execute(circ, backend, shots=100000).result().get_counts())
# {'11': 49823, '00': 50177}

# Changing the cat size from 16 (default) to 4 and k1/k2 to 1e-2.
backend = provider.get_backend(
    'EMU:6Q:PHYSICAL_CATS', average_nb_photons=4, kappa_2=1e4
)
print(execute(circ, backend, shots=100000).result().get_counts())
# {'01': 557, '11': 49422, '10': 596, '00': 49425}
```