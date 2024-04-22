def write_current_line(text: str) -> None:
    """Write a given text on the same line as the previous call of this
    function."""
    # We use a padding of 80 characters to be sure everything is flushed out
    # when we do the next call to this function.
    print(f'{text:<80}', end='\r', flush=True)
