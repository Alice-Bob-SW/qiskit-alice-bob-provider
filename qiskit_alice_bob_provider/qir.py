##############################################################################
# Copyright 2026 Alice & Bob
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
# The conversion rules of this file are ported from qiskit-qir
# (https://github.com/microsoft/qiskit-qir), under the MIT License:
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
##############################################################################

from typing import Callable, Dict, List, Sequence, Tuple

import pyqir
from pyqir import (
    BasicBlock,
    Builder,
    Constant,
    Context,
    Function,
    FunctionType,
    IntType,
    Linkage,
    PointerType,
    Value,
    const,
    entry_point,
    qis,
    rt,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Bit, Clbit, IfElseOp, Instruction

_GATES: Dict[str, Callable[..., None]] = {
    'ccx': qis.ccx,
    'cx': qis.cx,
    'cz': qis.cz,
    'h': qis.h,
    'measure': qis.mz,
    'reset': qis.reset,
    'rx': qis.rx,
    'ry': qis.ry,
    'rz': qis.rz,
    's': qis.s,
    'sdg': qis.s_adj,
    'swap': qis.swap,
    't': qis.t,
    'tdg': qis.t_adj,
    'x': qis.x,
    'y': qis.y,
    'z': qis.z,
}

_MICROSECONDS_PER_UNIT = {
    's': 1e6,
    'ms': 1e3,
    'us': 1,
    'ns': 1e-3,
    'ps': 1e-6,
}

_PREPARATIONS: Dict[str, Tuple[str, bool]] = {
    '0': ('z', False),
    '1': ('z', True),
    '+': ('x', False),
    '-': ('x', True),
}

_MEASUREMENTS = {'measure_x': 'mx'}


def circuit_to_qir(circuit: QuantumCircuit) -> str:
    """Convert a circuit to QIR.

    An instruction with no QIR intrinsic is replaced by its definition.

    Returns:
        The QIR module, as LLVM text.

    Raises:
        ValueError: If an instruction, a prepared state, an `if_else`
            condition or a delay unit has no QIR equivalent.
        RuntimeError: If PyQIR finds the produced module invalid.
    """
    module = pyqir.qir_module(Context(), circuit.name)
    _Emitter(module, circuit).emit()
    error = module.verify()
    if error is not None:
        raise RuntimeError(error)
    return str(module)


class _Emitter:
    def __init__(self, module: pyqir.Module, circuit: QuantumCircuit):
        self._module = module
        self._context = module.context
        self._circuit = circuit
        self._builder = Builder(self._context)
        self._qubits = {bit: i for i, bit in enumerate(circuit.qubits)}
        self._clbits = {bit: i for i, bit in enumerate(circuit.clbits)}
        self._functions: Dict[str, Function] = {}
        self._null = Constant.null(PointerType(IntType(self._context, 8)))

    def emit(self) -> None:
        circuit = self._circuit
        entry = entry_point(
            self._module, circuit.name, circuit.num_qubits, circuit.num_clbits
        )
        self._builder.insert_at_end(BasicBlock(self._context, 'entry', entry))
        rt.initialize(self._builder, self._null)
        for instruction in circuit.data:
            self._visit(
                instruction.operation, instruction.qubits, instruction.clbits
            )
        self._record_output()
        self._builder.ret(None)

    def _record_output(self) -> None:
        for register in self._circuit.cregs:
            size = const(IntType(self._context, 64), register.size)
            rt.array_record_output(self._builder, size, self._null)
            for bit in reversed(register):
                result = pyqir.result(self._context, self._clbits[bit])
                rt.result_record_output(self._builder, result, self._null)

    def _visit(
        self,
        operation: Instruction,
        qubits: Sequence[Bit],
        clbits: Sequence[Bit],
    ) -> None:
        name = operation.name
        qubit_values = [
            pyqir.qubit(self._context, self._qubits[bit]) for bit in qubits
        ]
        results = [
            pyqir.result(self._context, self._clbits[bit]) for bit in clbits
        ]
        if name in _GATES:
            _GATES[name](
                self._builder, *operation.params, *qubit_values, *results
            )
        elif name in _MEASUREMENTS:
            self._call(_MEASUREMENTS[name], [*qubit_values, *results])
        elif name == 'id':
            # Same as qiskit-qir, see
            # https://github.com/qir-alliance/pyqir/issues/74
            qis.x(self._builder, qubit_values[0])
            qis.x(self._builder, qubit_values[0])
        elif name == 'barrier':
            pass
        elif name == 'delay':
            self._delay(operation, qubit_values)
        elif name == 'initialize':
            self._initialize(operation, qubit_values)
        elif isinstance(operation, IfElseOp):
            self._if_else(operation, qubits, clbits)
        elif operation.definition is not None:
            self._visit_block(operation.definition, qubits, clbits)
        else:
            raise ValueError(
                f'The instruction "{name}" has no QIR equivalent and no '
                'definition.'
            )

    def _visit_block(
        self,
        block: QuantumCircuit,
        qubits: Sequence[Bit],
        clbits: Sequence[Bit],
    ) -> None:
        for instruction in block.data:
            self._visit(
                instruction.operation,
                [qubits[block.find_bit(q).index] for q in instruction.qubits],
                [clbits[block.find_bit(c).index] for c in instruction.clbits],
            )

    def _delay(self, operation: Instruction, qubits: Sequence[Value]) -> None:
        if operation.unit not in _MICROSECONDS_PER_UNIT:
            indices = [pyqir.qubit_id(qubit) for qubit in qubits]
            raise ValueError(
                f'The delay of {operation.duration} on qubits {indices} has '
                f'the unit "{operation.unit}", which QIR does not support. '
                'Give the delay duration in a time unit: s, ms, us, ns or ps.'
            )
        double = pyqir.Type.double(self._context)
        duration = (
            float(operation.duration) * _MICROSECONDS_PER_UNIT[operation.unit]
        )
        self._call('delay', [const(double, duration), *qubits])

    def _initialize(
        self, operation: Instruction, qubits: Sequence[Value]
    ) -> None:
        state = str(operation.params[0])
        if len(qubits) != 1 or state not in _PREPARATIONS:
            raise ValueError(
                f'Cannot prepare the state {operation.params} in QIR. '
                'Initialize one qubit at a time, in 0, 1, + or -.'
            )
        basis, is_excited = _PREPARATIONS[state]
        boolean = IntType(self._context, 1)
        self._call(f'prepare_{basis}', [const(boolean, is_excited), qubits[0]])

    def _if_else(
        self,
        operation: IfElseOp,
        qubits: Sequence[Bit],
        clbits: Sequence[Bit],
    ) -> None:
        condition = operation.condition
        if not isinstance(condition, tuple):
            raise ValueError(
                f'The if_else condition {condition} is not supported in QIR. '
                'Use a condition on one bit or on one register.'
            )
        target, value = condition
        bits = [target] if isinstance(target, Clbit) else list(target)
        if value >= 2 ** len(bits):
            raise ValueError(
                f'The value {value} is too large for {len(bits)} bits.'
            )
        values = [bool(value >> i & 1) for i in range(len(bits))]
        results = [pyqir.result(self._context, self._clbits[b]) for b in bits]

        def on_true() -> None:
            self._visit_block(operation.blocks[0], qubits, clbits)

        def on_false() -> None:
            if len(operation.blocks) > 1:
                self._visit_block(operation.blocks[1], qubits, clbits)

        self._branch(list(zip(results, values)), on_true, on_false)

    def _branch(
        self,
        conditions: List[Tuple[Value, bool]],
        on_true: Callable[[], None],
        on_false: Callable[[], None],
    ) -> None:
        if not conditions:
            on_true()
            return
        (result, is_one), rest = conditions[0], conditions[1:]

        def on_match() -> None:
            self._branch(rest, on_true, on_false)

        qis.if_result(
            self._builder,
            result,
            one=on_match if is_one else on_false,
            zero=on_false if is_one else on_match,
        )

    def _call(
        self,
        name: str,
        args: List[Value],
    ) -> None:
        function = self._functions.get(name)
        if function is None:
            function = Function(
                FunctionType(
                    pyqir.Type.void(self._context), [a.type for a in args]
                ),
                Linkage.EXTERNAL,
                f'__quantum__qis__{name}__body',
                self._module,
            )
            self._functions[name] = function
        self._builder.call(function, args)
