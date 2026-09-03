import sys
from typing import Any, Optional


class DocumentPortalException(Exception):
    def __init__(self, error_message: str, error_details: Optional[Any] = None):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_details = error_details

        # Extract traceback object safely
        exc_tb = None
        if error_details is not None and hasattr(error_details, "exc_info"):
            # Handles passing `sys` as error_details
            _, _, exc_tb = error_details.exc_info()
        elif isinstance(error_details, BaseException):
            # Handles passing an caught exception instance `e`
            exc_tb = error_details.__traceback__
        else:
            # Fallback to sys.exc_info()
            _, _, exc_tb = sys.exc_info()

        if exc_tb is not None:
            # Navigate to the innermost frame where the exception originated
            last_tb = exc_tb
            while last_tb.tb_next:
                last_tb = last_tb.tb_next
            self.file_name = last_tb.tb_frame.f_code.co_filename
            self.lineno = last_tb.tb_lineno
        else:
            self.file_name = "unknown"
            self.lineno = 0

    def __str__(self) -> str:
        return (
            f"DocumentPortalException occurred in file [{self.file_name}] "
            f"at line [{self.lineno}] with message: {self.error_message}"
        )

    def __repr__(self) -> str:
        return f"DocumentPortalException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"