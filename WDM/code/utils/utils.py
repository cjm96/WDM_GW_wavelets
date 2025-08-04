def next_multiple(i: int, N: int) -> int:
    """
    Return smallest integer multiple of N greater than or equal to integer i.
    
    Parameters
    ----------
    i : int
        The input number.
    N : int
        The multiple to align to.

    Returns
    -------
    j : int
        The next multiple of N.

    Notes
    -----
    Example with N = 3:

    >>> for i in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
    ...     print(f"{i} -> {next_multiple(i, 4)}")
    -4 -> -3
    -3 -> -3
    -2 ->  0
    -1 ->  0
     0 ->  0
     1 ->  3
     2 ->  3
     3 ->  3
     4 ->  6
    """
    j = ((i + N - 1) // N) * N
    return j


def C_nm(n: int, m: int) -> complex:
    """
    Compute the complex-valued modulation coefficient :math:`C_{nm}`.

    This coefficient alternates between 1 and :math:`i` to apply modulation
    in the WDM transform.

    Parameters
    ----------
    n : int
        Time index.
    m : int
        Frequency index.

    Returns
    -------
    complex
        Coefficient :math:`C_{nm}`, equal to 1 or :math:`i` depending on 
        parity of :math:`n+m`.
    """
    return 1.0 if (n + m) % 2 == 0 else 1.0j