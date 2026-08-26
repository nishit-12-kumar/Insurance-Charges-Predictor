"""Custom exception wrapper: every raised error gets the file name, line
number, and original message attached, for debuggable production logs.
"""
from __future__ import annotations

import sys


def _build_error_message(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error: {error}"
    return (
        f"Error occurred in script [{exc_tb.tb_frame.f_code.co_filename}] "
        f"at line number [{exc_tb.tb_lineno}] with message: [{error}]"
    )


class InsuranceCostException(Exception):
    """Base exception for all custom errors in this project.

    Usage:
        try:
            ...
        except Exception as e:
            raise InsuranceCostException(e, sys) from e
    """

    def __init__(self, error: Exception, error_detail: sys = sys) -> None:
        super().__init__(str(error))
        self.error_message = _build_error_message(error, error_detail)

    def __str__(self) -> str:
        return self.error_message
