# Alice & Bob Qiskit provider

This project contains a provider that allows access to
[Alice & Bob](https://alice-bob.com/) QPUs from the Qiskit framework.

## Installation

You can install the provider using |`pip`:

```bash
pip install qiskit-alice-bob-provider
```

`pip` will handle installing all the python dependencies automatically and you
will always install the  latest (and well-tested) version.

## Use your Alice & Bob API key

You can initialize the Alice & Bob provider using your API key locally with:

```python
from qiskit_alice_bob_provider import AliceBobProvider
ab = AliceBobProvider('MY_API_KEY')
```

Where `MY_API_KEY` is your API key to the Alice & Bob API.

```python
print(ab.backends())
backend = ab.get_backend('SINGLE_CAT_SIMULATOR')
```

The backend can then be used like a regular Qiskit backend:

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
