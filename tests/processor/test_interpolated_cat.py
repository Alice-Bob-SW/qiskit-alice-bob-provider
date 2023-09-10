import numpy as np

from qiskit_alice_bob_provider.processor.interpolated_cat import (
    InterpolatedCatProcessor,
)
from qiskit_alice_bob_provider.processor.serialization.model import (
    SerializedProcessor,
)


def test_apply_instruction() -> None:
    with open(
        'tests/processor/serialization/data/all_types.json',
        encoding='utf-8',
    ) as f:
        ser = SerializedProcessor.model_validate_json(f.read())
    proc = InterpolatedCatProcessor(ser, alpha=np.sqrt(5))
    applied = proc.apply_instruction('delay', (0,), [500])
    assert applied.quantum_errors is not None
    assert applied.readout_errors is None
    applied = proc.apply_instruction('mx', (0,), [])
    assert applied.quantum_errors is not None
    assert applied.readout_errors is not None


def test_all_instructions() -> None:
    with open(
        'tests/processor/serialization/data/all_types.json',
        encoding='utf-8',
    ) as f:
        ser = SerializedProcessor.model_validate_json(f.read())
    proc = InterpolatedCatProcessor(ser, alpha=np.sqrt(5))
    list(proc.all_instructions())
    instr = next(i for i in proc.all_instructions() if i.name == 'mx')
    assert instr.readout_errors is not None
