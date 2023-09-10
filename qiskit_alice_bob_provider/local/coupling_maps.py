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

from typing import List, Tuple


def bidirect_map(map: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """From a coupling map where edges flow in one direction only, add
    edges pointing in the opposite direction."""
    out = []
    for i, j in map:
        out.append((i, j))
        out.append((j, i))
    return out


def circular_map(n_qubits: int) -> List[Tuple[int, int]]:
    """A bidirected coupling map where qubits form a circle"""
    return bidirect_map([(i, (i + 1) % n_qubits) for i in range(n_qubits)])


def _rect_index(width: int, i: int, j: int) -> int:
    return i * width + j


def rectangular_map(height: int, width: int) -> List[Tuple[int, int]]:
    """A bidrected coupling map where qubits are aligned on a grid"""
    map = []
    for i in range(height):
        for j in range(width):
            here = _rect_index(width, i, j)
            if i + 1 < height:
                below = _rect_index(width, i + 1, j)
                map.append((here, below))
            if j + 1 < width:
                right = _rect_index(width, i, j + 1)
                map.append((here, right))
    return bidirect_map(map)
