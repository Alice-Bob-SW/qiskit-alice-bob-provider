##############################################################################
# Copyright 2023 Alice & Bob
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

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from qiskit.circuit import Delay, Instruction, Measure, Parameter, Reset
from qiskit.circuit.library import (
    Barrier,
    CCXGate,
    CXGate,
    CZGate,
    HGate,
    Initialize,
    RXGate,
    RYGate,
    RZGate,
    RZZGate,
    SGate,
    SwapGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.transpiler import Target

from ..custom_instructions import MeasureX


def ab_target_to_qiskit_target(ab_target: Dict) -> Target:
    """From a description of a target served by the Alice & Bob API, extract
    a Qiskit transpilation target.

    Args:
        ab_target (dict): the object returned by the Alice & Bob API and
            describing a target

    Returns:
        Target: a Qiskit target
    """
    instructions: Dict[str, Instruction] = {}
    for ab_instr in ab_target['instructions']:
        qiskit_instrs = _qir_signature_to_qiskit_instructions(
            ab_instr['signature']
        )
        for name, instr in qiskit_instrs:
            instructions[name] = instr
    target = Target(num_qubits=ab_target['numQubits'])
    for name, instr in instructions.items():
        target.add_instruction(instr, name=name)
    return target


_UNSUPPORTED_WARNING = (
    'The Alice & Bob API supports a function signature "{}" that is not '
    'recognized by this Qiskit provider'
)


# pylint: disable=too-many-branches,too-many-return-statements
def _qir_signature_to_qiskit_instructions(
    signature: str,
) -> List[Tuple[str, Instruction]]:
    """From a QIR signature, determine the corresponding Qiskit instructions"""
    parsed_instr = _parse_signature(signature)
    instr_short_name = _parse_function_name(parsed_instr.name)
    if instr_short_name is None:
        warn(_UNSUPPORTED_WARNING.format(signature))
        return []
    elif instr_short_name == 'read_result':
        # For reasons outside the scope of this provider, the API must return
        # this function in the list of supported instructions.
        # Since this function is not supported by Qiskit, we just ignore it.
        return []
    elif instr_short_name == 'barrier':
        return [('barrier', Barrier(0))]
    elif instr_short_name in {'ccx', 'toffoli'}:
        return [('ccx', CCXGate())]
    elif instr_short_name in {'cx', 'cnot'}:
        return [('cx', CXGate())]
    elif instr_short_name == 'cz':
        return [('cz', CZGate())]
    elif instr_short_name == 'h':
        return [('h', HGate())]
    elif instr_short_name in {'mz', 'm', 'measure'}:
        return [('measure', Measure())]
    elif instr_short_name == 'reset':
        return [('reset', Reset())]
    elif instr_short_name == 'delay':
        duration = Parameter('t')
        return [('delay', Delay(duration, unit='us'))]
    elif instr_short_name == 'prepare_z':
        return [('initialize', Reset()), ('initialize', Initialize([0, 1]))]
    elif instr_short_name == 'prepare_x':
        return [
            ('initialize', Initialize([1 / np.sqrt(2), 1 / np.sqrt(2)])),
            ('initialize', Initialize([1 / np.sqrt(2), -1 / np.sqrt(2)])),
        ]
    elif instr_short_name == 'rx':
        phi = Parameter('ϕ')
        return [('rx', RXGate(phi))]
    elif instr_short_name == 'ry':
        phi = Parameter('ϕ')
        return [('ry', RYGate(phi))]
    elif instr_short_name == 'rz':
        phi = Parameter('ϕ')
        return [('rz', RZGate(phi))]
    elif instr_short_name == 's':
        return [('s', SGate())]
    elif instr_short_name == 'swap':
        return [('swap', SwapGate())]
    elif instr_short_name == 't':
        return [('t', TGate())]
    elif instr_short_name == 'tdg':
        return [('tdg', TdgGate())]
    elif instr_short_name == 'x':
        return [('x', XGate())]
    elif instr_short_name == 'y':
        return [('y', YGate())]
    elif instr_short_name == 'z':
        return [('z', ZGate())]
    elif instr_short_name == 'mx':
        return [('measure_x', MeasureX())]
    elif instr_short_name == 'rzz':
        phi = Parameter('ϕ')
        return [('rzz', RZZGate(phi))]
    warn(_UNSUPPORTED_WARNING.format(signature))
    return []


@dataclass
class _QirFunction:
    name: str
    return_type: str
    arguments: List[str]


def _parse_signature(signature: str) -> _QirFunction:
    name, types = signature.split(':', maxsplit=1)
    name = name.strip()
    if '(' not in types:
        return _QirFunction(name=name, return_type=types.strip(), arguments=[])
    return_type, arguments_str = types.split('(', maxsplit=1)
    return_type = return_type.strip()
    arguments_str = arguments_str.rstrip(')')
    arguments: List[str] = []
    for arg in arguments_str.split(','):
        arguments.append(arg.strip())
    return _QirFunction(
        name=name, return_type=return_type, arguments=arguments
    )


_QIS_FUNCTION_PATTERN = r'__quantum__qis__([a-z0-9_]+)__(body|adj)'


def _parse_function_name(name: str) -> Optional[str]:
    m = re.search(_QIS_FUNCTION_PATTERN, name)
    if m is None:
        return None
    call_name = m.group(1)

    # QIR uses the pattern "adj" instead of "body" for the adjoint gates
    # (such as sdg, tdg).
    if m.group(2) == 'adj':
        call_name += 'dg'
    return call_name
