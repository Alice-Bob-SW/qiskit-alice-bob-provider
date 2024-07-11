from datetime import datetime

from IPython import get_ipython
from IPython.display import display

# There is no stubs for the ipywidget library
from ipywidgets import Label  # type: ignore

current_label = Label()


def display_new_line() -> None:
    """Append a new line to display content."""
    if is_ipython():
        _new_line_ipython()
    else:
        _new_line_std()


def display_current_line(text: str) -> None:
    """Replace text on the current line with the current input."""
    if is_ipython():
        _display_ipython(text)
    else:
        _display_std(text)


def _new_line_ipython() -> None:
    # pylint: disable=global-statement, invalid-name
    global current_label
    current_label = Label()
    display(current_label)


def _display_ipython(text: str) -> None:
    current_label.value = f'[{datetime.now().strftime("%H:%M:%S")}] {text}'


def _new_line_std() -> None:
    print()


def _display_std(text: str) -> None:
    # We use a padding of 80 characters to be sure everything is flushed out
    # when we do the next call to this function.
    print(
        f'[{datetime.now().strftime("%H:%M:%S")}] {text:<80}',
        end='\r',
        flush=True,
    )


def is_ipython() -> bool:
    """Return true if we are from an ipython environment"""
    try:
        shell = get_ipython()
        if shell is None:
            return False
        return True
    except ModuleNotFoundError:
        return False
