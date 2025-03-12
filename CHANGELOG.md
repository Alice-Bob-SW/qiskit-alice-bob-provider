# CHANGELOG


## v1.1.2-beta.2 (2025-03-12)

### Build System

- **deps**: Upgrade qiskit-aer to <0.17
  ([`f672e2a`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/f672e2ad28848c01bb131085d15512abbb6e0c1d))


## v1.1.2-beta.1 (2025-02-14)

### Bug Fixes

- Issue when options are passed as numpy types
  ([`c1523cd`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/c1523cd414dd3b28bcd53caf1778ef07a6c97ab9))

When using the AliceBobRemoteProvider, the backend options are serialized as JSON to be sent to the
  API. In some cases, the user might provide some option values as numpy types (eg.
  `average_nb_photons=np.int32(4)`), which is not JSON serializable and therefore raises an error.

This fix aims to convert numpy scalar values to their corresponding native Python type before
  serialization. `value.item()` does just that.


## v1.1.1 (2025-02-13)

### Bug Fixes

- Add missing native instructions for logical targets
  ([`4fe3ad0`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/4fe3ad04d015a5c99dece81e8b7021f3f9da7bce))

Our logical processors are expected to support the S (and S adj.) gates in their basis gate set.

This commits add them to the basis gate set, so that we don't try to transpile them.

- Allow transpilation of circuits multiple qregs.
  ([`d06f2f6`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/d06f2f624934c887db3c5785527bef9a868e7c70))

We were not properly mapping the virtual to physical qubits in our custom pass manager, which
  resulted in the impossibility of using the full capabilities of our backends.

Indeed, this prevented from defining circuits with multiple quantum registries that Qiskit would
  have been able to a single physical registry to run on.

This commit addresses that, adding a Layout when applying the custom pass manager of our backends to
  ensure we can map qubits to physical registries.

- Measure_x custom instruction transpiling
  ([`1aecb2f`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/1aecb2f2bf85c4ebb9263eda3cd4eda1b868ddc9))

The function substitute_node_with_dag is patched to make the custom measure_x instruction work with
  Qiskit 1.2. It adds the else statement that was removed on lines 1610-1611 in dagcircuit.py in
  this commit :
  https://github.com/Qiskit/qiskit/commit/353b0ea6bdd907e801ad8fa264f3444e0be942aa#diff-4fb31a3ade5ae57cfd91ea00dbf3c5b6ab066a8234a742d91f9c09a09edca2f7L1610-L1611
  This function will be moved to Rust in Qiskit version 1.3 so we will have to find another
  solution.

- Migrate deprecated pkg_resources to importlib.metadata
  ([`7c87c70`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/7c87c7048baf80c1b6226625b99fcf1b38d640c8))

Refs: #7228

- Migrate numpy.infty for 2.0 compatibility
  ([`f5b4a0a`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/f5b4a0a35b21c9445f5d4d563efde7e4a5022edb))

Numpy deprecated the term 'infty' and now only allows the value of 'inf'

- Prevent rounding inequalities with close instead of equal
  ([`cba9f35`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/cba9f355e09eb306822c7a84808fb6f857400cb4))

- Set Python versions >= 3.8 <3.12
  ([`249b719`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/249b719f836aecf3da9e69e5cbfd7c5af65ea45b))

- Skip reset for the SK-Synthesis
  ([`3c1f7d3`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/3c1f7d3d3f47ae793c7570fc8f39581008f96a0e))

This bug was highlighted after the update to the transpilation pass. It appears the remote target
  including the logical ones include the reset gate in their list of accepted input gates.

This commit makes it so we don't try to compute approximations for the reset gate.

- Too-many-positional-arguments linting
  ([`9784f68`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/9784f68b6a89b6e1b5ef448e68730e7bacdd83d2))

Refs: #7228

- Transpilation for targets with only Clifford + T gate basis
  ([`43bee3d`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/43bee3d6ed84ac92135defdacc9dbe11a747cd26))

- **local**: Do not allow to pass A&B custom options to backend.run()
  ([`0e9b8cf`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/0e9b8cfea3b68f741fb4bb3d4cbdfbf84771518f))

The Alice & Bob custom options of Processors like average_nb_photons are not allowed for
  backend.run() with local provider because they will be ignored. They should be passed to
  get_backend() instead.

- **local**: Return new backend instances instead of shared references
  ([`8d48b4d`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/8d48b4d2cef5378d1489e2d578ada8015e87ba6a))

These changes are made to fix shared references issues in local provider and harmonize the behavior
  with remote provider.

- **remote**: Backend default options not set properly in remote provider
  ([`3c7268e`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/3c7268e4e3b8acb72550c956e2cb7e0ca497b6f5))

When using AliceBobRemoteProvider.get_backend() with parameters, a Python reference of the backend
  was used and the parameters were modified on the object of the AliceBobRemoteProvider instance
  created earlier. It was causing the default options of a backend to be overwritten definitely for
  an AliceBobRemoteProvider instance when calling get_backend() with parameters.

These changes use a new instance of the backend object instead of a reference to the same instance
  to make each backend returned by get_backend() and backends() totally independent (as
  get_backends() uses backends() under the hood). We need to recreate the backend instance each time
  and copy.deepcopy() was not used to avoid any potential issues with cyclic graph or things like
  that.

- **remote**: Block non QuantumCircuit experiments
  ([`9b5f91a`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/9b5f91af48c002445086a690d62657f92d670c77))

- **remote**: Shared reference of options in backend.run()
  ([`7355663`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/7355663fcd09f545521abcec9a4b4d0f8a95a037))

A new instance of Options is needed to allow to modify the options temporarily. We need to call
  Options.update_options() without definitely modifying the options of the backend through a shared
  reference.

Some code is factorized under update_options_object() to allow to update an Options object different
  from the AliceBobRemoteBackend.options attribute.

- **remote**: Support the TDG gate
  ([`574a043`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/574a043da52ade66d97bb06df4abe1e609b04633))

- **sk_synthesis**: Ignore cswap gates in synthesis
  ([`8fde3db`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/8fde3db72d439a320703c4ff695c96612e7cea76))

- **sk_synthesis**: Remove cz & ccz from synth gates
  ([`9a6dac1`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/9a6dac1e138d5bd1bc548fb9345ada39eb6b32dd))

We used to apply the SK synthesis algorithm to the CZ & CCZ gates. This behaviour was inappropriate
  as these gates are discrete gates, and should not be approximated like a continuous one (e.g.:
  rZ(theta)).

The CZ & CCZ should rather be replaced by CX & CCX respectively in the "Basis Translation" step of
  transpilation instead.

This commit does this by removing CZ & CCZ from the list of gates to synthesize, so they will be
  ignored by the SK synthesis step.

- **state_preparation**: Unroll custom def. before synthesis
  ([`9f1656c`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/9f1656ce7f2700513bdc99480295bbfdd27600cc))

We are missing this step to be able to transpile circuits with custom defined gates (such as
  Margolus). The gates are now unrolled to their lowest definitions before being synthetized.

### Documentation

- Add warning in readme for macOS transpilation error
  ([`f9639f8`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/f9639f8c876fbe5d1bbf42d91e55eec36e86e6df))

- **readme**: Update readme example code to provider v1.0
  ([`ffb22d5`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/ffb22d5be9c053d6dbb5fceea4d5e65ddb73f18b))

### Features

- Add get_memory access to the AliceBobRemoteProvider
  ([`776d965`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/776d965f49f89541662044e17f0d1216a9ae74f5))

- Add verbose mode for Remote provider
  ([`0988a53`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/0988a5333ef53a7e70fd8ff276139acc14e6fa2a))

- Alert users on outdated provider
  ([`0a69337`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/0a69337f476b8b2fbbc19b45fc53acc041bb8f2c))

- Pre-release 1.0.0rc0
  ([`c5847cb`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/c5847cbf31880f214b8467432207eb665c832778))

- Release 0.5.1
  ([`3cc3a0a`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/3cc3a0a0a202f6e6f08c321c39a403f946bdaea2))

- Release 0.5.2
  ([`0709e4f`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/0709e4fe34b3180725bd045d8235a11d5826af80))

- Release 0.5.3
  ([`a5b8147`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/a5b8147bdb50f45e956bea57abccd9ebcbb39477))

- Release 1.0.0
  ([`5add8d1`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/5add8d126ce88463fa4fe5e09f961633ceabde72))

- Release 1.0.0rc1
  ([`ae30249`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/ae30249cf4d74404e77b2b2ca6945fcc61d98374))

- Release 1.0.0rc2
  ([`2217f76`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/2217f76e9e8d6ef95c1a9387efc41ebb9f4f2ad0))

- Release 1.0.1
  ([`99c98be`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/99c98bebb85bdb20756e160599e75671e23efe15))

- Release version 0.5.4
  ([`7c804ef`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/7c804ef63b294ae8318e445f53030055fa53f168))

- Release version 0.6.0
  ([`801e08f`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/801e08f98bd0890485b7a97db045058a482bf4c2))

- Release version 0.7.0
  ([`7839b44`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/7839b44e75667e670ed5709a50e5ea6757f2009e))

- Release version 0.7.1
  ([`daafc1e`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/daafc1ea988979d63f97ce3ef00e0cb2a9474b5a))

- Release version 0.7.2
  ([`da27b72`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/da27b72b73af4358433bd1f98991ee53ebcb0be1))

- Release version 1.1.0
  ([`21a7ffc`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/21a7ffc122191c6074e16a6c78fadb9bfa3bcc40))

- Support qiskit 1.0
  ([`5efbf76`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/5efbf76f35e2cb8ea1e17d44c04fdb1911fc27b4))

- Update packages for python 3.12 support
  ([`aef427d`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/aef427d90e975a9e5f13a945c00d702a6e2e72ab))

Refs: #7228

- Update qiskit to v1.2
  ([`7ebc05a`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/7ebc05a44014ea1c2a6a5cd4280b88f1f858043b))

- Update to qiskit 1.3
  ([`7034b93`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/7034b9326753d84e602e989e93e25091e819fe02))

Some tests had to be updated for Qiskit 1.3. This new Qiskit version changes the default
  optimization_level of transpile() from 1 to 2. This change enables other transpilation passes like
  VF2Layout and ApplyLayout passes that shuffle the qbits randomly in the circuit. As we need a
  deterministic and reproducible behavior for the tests, we fix the seed in transpile().

- Use ipywidget to display job status on notebooks
  ([`5acc856`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/5acc8566844b63f5cf1c1944c5d68cae3fba11a6))

- **gitignore**: Add debug notebook
  ([`87a1492`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/87a149245e551437d219dae3ae51f4c82bb0febb))

- **remote**: Add basic monitoring of job execution
  ([`120cace`](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/commit/120cacea3bb5fc1f855365d89abf8fabee7929a2))
