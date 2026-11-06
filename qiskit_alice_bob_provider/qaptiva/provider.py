import json
from typing import Dict

from qat.comm.exceptions.ttypes import QPUException
from qiskit.exceptions import QiskitError

from ..remote.provider import AliceBobTargetProvider
from .backend import AliceBobQaptivaBackend

# The version of the job contract with anb-myqlm-server.
PROTOCOL_VERSION = '1'


class AliceBobQaptivaProvider(AliceBobTargetProvider[AliceBobQaptivaBackend]):
    """
    Class listing and providing access to the Alice & Bob backends reached
    through an anb-myqlm-server, for example from a Qaptiva node.
    """

    def __init__(self, remote_qpu):
        """
        Args:
            remote_qpu: a myQLM QPU that reaches the anb-myqlm-server, such
                as ``RemoteQPU(port=1235, ip='10.0.0.1')``, with or without
                plugins in front of it.
        """
        if not all(
            hasattr(remote_qpu, method) for method in ('get_specs', 'submit')
        ):
            raise TypeError(
                'remote_qpu must be a myQLM QPU, with get_specs and submit, '
                f'got {type(remote_qpu).__name__}'
            )
        self._remote_qpu = remote_qpu
        try:
            meta_data = remote_qpu.get_specs().meta_data or {}
        except QPUException as exc:
            raise QiskitError(
                f'The myQLM server failed: {exc.message}'
            ) from exc
        version = meta_data.get('protocol_version')
        if version != PROTOCOL_VERSION:
            raise QiskitError(
                f'The myQLM server has protocol version {version}, and this '
                f'provider needs version {PROTOCOL_VERSION}. Use the '
                'anb-myqlm-server release that matches this provider.'
            )
        self._targets = json.loads(meta_data['targets'])

    def _make_backend(self, ab_target: Dict) -> AliceBobQaptivaBackend:
        return AliceBobQaptivaBackend(self._remote_qpu, ab_target)
