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

"""Convenience transpile wrapper that auto-detects the optimization stage
plugin from Alice & Bob backends.

Qiskit's transpile() does not auto-detect get_optimization_stage_plugin()
from the backend (only translation and scheduling are auto-detected). This
wrapper fills that gap.
"""

from qiskit import transpile as qiskit_transpile


def transpile(circuits, backend, **kwargs):
    """Transpile circuits for the given backend, auto-detecting the
    optimization stage plugin if available.

    This is a thin wrapper around qiskit.transpile() that checks whether the
    backend provides a custom optimization stage plugin via
    get_optimization_stage_plugin(), and passes it as optimization_method
    if the caller hasn't already specified one.

    Args:
        circuits: Circuit(s) to transpile.
        backend: Backend to transpile for.
        **kwargs: All other arguments are forwarded to qiskit.transpile().

    Returns:
        The transpiled circuit(s).
    """
    if 'optimization_method' not in kwargs:
        if hasattr(backend, 'get_optimization_stage_plugin'):
            plugin = backend.get_optimization_stage_plugin()
            if plugin is not None:
                kwargs['optimization_method'] = plugin
    return qiskit_transpile(circuits, backend, **kwargs)
