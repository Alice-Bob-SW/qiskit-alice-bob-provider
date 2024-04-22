import logging
from enum import Enum

from qiskit.providers import JobStatus


class AliceBobEventType(Enum):
    CREATED = 'CREATED'
    FETCHING_INPUT = 'FETCHING_INPUT'
    INPUT_READY = 'INPUT_READY'
    COMPILING = 'COMPILING'
    COMPILED = 'COMPILED'
    COMPILATION_FAILED = 'COMPILATION_FAILED'
    TRANSPILING = 'TRANSPILING'
    TRANSPILED = 'TRANSPILED'
    TRANSPILATION_FAILED = 'TRANSPILATION_FAILED'
    EXECUTING = 'EXECUTING'
    SUCCEEDED = 'SUCCEEDED'
    EXECUTION_FAILED = 'EXECUTION_FAILED'
    CANCELLED = 'CANCELLED'
    TIMED_OUT = 'TIMED_OUT'
    UNKNOWN = 'UNKNOWN'

    @classmethod
    def _missing_(cls, value):
        logging.warning(
            f'Received unexpected job event {value}. \n'
            'Please ensure you are running the latest version of '
            'the Alice & Bob Provider.'
        )
        return cls.UNKNOWN

    # pylint: disable=too-many-return-statements
    def to_qiskit_status(self) -> JobStatus:
        """_summary_
        Args:
            event (str): an event from the Alice & Bob API, usually the latest
                event about a given job

        Returns:
            JobStatus: a Qiskit job status
        """
        if self == AliceBobEventType.CREATED:
            return JobStatus.INITIALIZING
        elif self in {
            AliceBobEventType.INPUT_READY,
            AliceBobEventType.COMPILED,
            AliceBobEventType.TRANSPILED,
        }:
            return JobStatus.QUEUED
        elif self in {
            AliceBobEventType.COMPILING,
            AliceBobEventType.TRANSPILING,
        }:
            return JobStatus.VALIDATING
        elif self == AliceBobEventType.EXECUTING:
            return JobStatus.RUNNING
        elif self in {
            AliceBobEventType.COMPILATION_FAILED,
            AliceBobEventType.TRANSPILATION_FAILED,
            AliceBobEventType.EXECUTION_FAILED,
            AliceBobEventType.TIMED_OUT,
        }:
            return JobStatus.ERROR
        elif self == AliceBobEventType.SUCCEEDED:
            return JobStatus.DONE
        elif self == AliceBobEventType.CANCELLED:
            return JobStatus.CANCELLED
        # An unknown job status will be considered to be an Error by default.
        return JobStatus.ERROR
