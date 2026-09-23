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

The provider requires Python 3.10 to 3.13 and Qiskit 2.x (2.5 or a later 2.x
release).

## Migration from Qiskit 1.x

If your code was written for Qiskit 1.x, update it as follows:

- Replace `c_if` with `if_test`:

  ```python
  # Qiskit 1.x
  circ.x(0).c_if(0, 1)
  # Qiskit 2.x
  with circ.if_test((circ.clbits[0], 1)):
      circ.x(0)
  ```

- Read the result headers as dictionaries: write
  `result.results[0].header['name']`, not `result.results[0].header.name`.
- `qiskit.execute`, `BackendV1` and `ProviderV1` do not exist. Call
  `transpile(circ, backend)`, then `backend.run(...)`. The provider backends
  are `BackendV2` instances.
- Give the backend to `transpile`. `transpile(circ, basis_gates=[...])` does
  not accept the provider instructions (`initialize`, `measure_x`), and
  `transpile` has no `instruction_durations` argument.
- Bind the parameters with `assign_parameters` before you call `transpile`.
  The durations of `delay` and `rz` and the Solovay-Kitaev synthesis need the
  parameter values, so an unbound parameter causes an error. For the same
  reason, do not give parameter values to `BackendSamplerV2` in a PUB.
- `QuantumCircuit.duration` still works, but Qiskit marks it as deprecated.
  Its replacement, `QuantumCircuit.estimate_duration(backend.target)`, does
  not work with the provider backends yet.
- On `EMU:40Q:PHYSICAL_CATS`, prepare each qubit with its own `initialize`.
  A multi-qubit `initialize`, for example `circ.initialize('0000+')`, stops the
  VF2 layout pass, and the transpiler can then add SWAP gates.

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
# EMU:6Q:PHYSICAL_CATS, EMU:40Q:PHYSICAL_CATS, EMU:40Q:LOGICAL_TARGET,
# EMU:40Q:LOGICAL_NOISELESS, EMU:15Q:LOGICAL_EARLY, EMU:1Q:LESCANNE_2020
```

The `EMU:nQ:PHYSICAL_CATS` backends are theoretical models of quantum processors made
up of physical cat qubits.
They can be used to study the properties of error correction codes implemented
with physical cat qubits, for different hardware performance levels
(see the parameters of class `PhysicalCatProcessor`).

The `EMU:nQ:LOGICAL_*` backends model logical qubits made of physical cat
qubits, assembled with a repetition code that corrects phase flips.
They expose the discrete Clifford+T gate set (`h`, `s`, `sdg`, `t`, `tdg`, `x`,
`z`, `cx`, `ccx`), so the transpiler approximates any other rotation with the
Solovay-Kitaev algorithm.
The default recursion degree is 3, and the default depth of the basic
approximations is 5. A higher degree gives a longer circuit. It gives a more
accurate approximation only if the depth is also high enough. To change them:

```python
transpile(
    circ,
    backend,
    unitary_synthesis_plugin_config={'recursion_degree': 3, 'depth': 10},
)
```

Qiskit 2.5 does not accept a `basic_approximations` value in this
configuration.

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
# {'00': 49910, '11': 50090}

# Changing the cat size from 16 (default) to 4 and k1/k2 to 1e-2.
backend = provider.get_backend(
    'EMU:6Q:PHYSICAL_CATS', average_nb_photons=4, kappa_2=1e4
)
print(backend.run(circ, shots=100000).result().get_counts())
# {'01': 1788, '10': 1757, '00': 48122, '11': 48333}
```

## Setting Up Development Environment (for contributors only)

You need [uv](https://docs.astral.sh/uv/). Then run:

```bash
uv sync                        # create .venv/ with dev dependencies
uv run pre-commit install      # install the git hooks
uv run pytest                  # run the tests
uv run ruff format             # format the code
uv run ruff check              # lint the code
uv run mypy .                  # check the types
```

Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/): release version
numbers are computed from them.
