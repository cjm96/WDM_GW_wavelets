import doctest

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import WDM
import WDM.code.utils.utils

utils = WDM.code.utils.utils


def test_check_working_set_allows_and_refuses():
    r"""
    Test that the working-set guard permits what fits and refuses what does
    not.

    The limit is pinned here rather than detected. If it were left to
    `detect_memory_bytes` the outcome would depend on the machine running
    the tests: the refusal branch would go unexercised on a large node, and
    the permitted branch could fail on a small laptop.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(utils, "detect_memory_bytes", lambda: 1 << 30)
        patch.setattr(utils, "MAX_WORKING_SET_FRACTION", 0.25)

        limit = 1 << 28  # a quarter of the 1 GiB we just pinned

        assert utils.check_working_set(limit - 1, "an array which",
                                       "fast_method") is None, \
            "The guard should permit a working set below the limit."

        assert utils.check_working_set(limit, "an array which",
                                       "fast_method") is None, \
            "The guard should permit a working set exactly at the limit."

        with pytest.raises(MemoryError):
            utils.check_working_set(limit + 1, "an array which",
                                    "fast_method")


def test_check_working_set_message_is_actionable():
    r"""
    Test that the refusal explains itself.

    The message is the whole point of the guard: refusing without saying how
    much was wanted, how much is allowed, what to call instead and how to
    lift the limit would only move the confusion somewhere else.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(utils, "detect_memory_bytes", lambda: 1 << 30)
        patch.setattr(utils, "MAX_WORKING_SET_FRACTION", 0.25)

        with pytest.raises(MemoryError) as excinfo:
            utils.check_working_set(1 << 34,
                                    "Gnm_basis builds an array which",
                                    "forward_transform_fft")

        message = str(excinfo.value)

    for expected in ["Gnm_basis",                  # what was being built
                     "16.00 GiB",                  # how much it wanted
                     "256.00 MiB",                 # how much is allowed
                     "forward_transform_fft",      # what to call instead
                     "MAX_WORKING_SET_FRACTION"]:  # how to lift the limit
        assert expected in message, \
            f"The refusal message should mention {expected!r}, got {message!r}"


def test_detect_memory_bytes():
    r"""
    Test that the detected memory size is a plausible number of bytes.

    The exact value is a property of the machine, so only the magnitude can
    be asserted portably.
    """
    nbytes = utils.detect_memory_bytes()

    assert isinstance(nbytes, int), \
        f"detect_memory_bytes should return an int, got {type(nbytes)}."

    assert 1 << 26 < nbytes < 1 << 50, \
        f"Detected memory of {utils.format_bytes(nbytes)} is not plausible."


def test_detect_memory_bytes_falls_back():
    r"""
    Test the fallback used when the machine cannot be interrogated.

    This is the Windows path - `os.sysconf` does not exist there - so it
    would never run on the platforms we test on unless forced.
    """
    def unavailable(*args, **kwargs):
        raise OSError("unavailable")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(utils.jax, "devices", unavailable)
        patch.setattr(utils.os, "sysconf", unavailable)

        assert utils.detect_memory_bytes() \
                    == utils.MAX_WORKING_SET_BYTES_FALLBACK, \
            "Undetectable memory should fall back to the default limit."


def test_format_bytes_boundaries():
    r"""
    Test the unit boundaries of the byte formatter.

    The docstring example covers exact powers of 1024; the interesting cases
    are either side of a boundary, and the largest unit, which has to absorb
    everything above it.
    """
    cases = [(0, "0.00 B"),
             (1023, "1023.00 B"),
             (1 << 10, "1.00 KiB"),
             (1 << 20, "1.00 MiB"),
             (1 << 30, "1.00 GiB"),
             (1 << 40, "1.00 TiB"),
             (1 << 50, "1024.00 TiB")]

    for nbytes, expected in cases:
        assert utils.format_bytes(nbytes) == expected, \
            f"format_bytes({nbytes}) should be {expected!r}, " \
            f"got {utils.format_bytes(nbytes)!r}."


def test_C_nm():
    r"""
    Test the modulation coefficient, which is 1 or i by the parity of n+m.
    """
    assert utils.C_nm(0, 0) == 1.0 and utils.C_nm(1, 1) == 1.0, \
        "C_nm should be 1 when n+m is even."

    assert utils.C_nm(0, 1) == 1.0j and utils.C_nm(1, 0) == 1.0j, \
        "C_nm should be i when n+m is odd."

    for n in range(4):
        for m in range(4):
            assert utils.C_nm(n, m) == (1.0 if (n + m) % 2 == 0 else 1.0j), \
                f"C_nm({n}, {m}) has the wrong parity."


def test_docstring_examples():
    r"""
    Run the worked examples in the utils docstrings.

    Nothing executed these before, so they had drifted away from what the
    functions actually return.
    """
    results = doctest.testmod(WDM.code.utils.utils, verbose=False)

    assert results.attempted > 0, \
        "No docstring examples were found to run."

    assert results.failed == 0, \
        f"{results.failed} of {results.attempted} docstring examples failed."


def test_x64():
    r"""
    Test that the WDM module is using float64 precision.
    """
    assert jax.config.read("jax_enable_x64"), \
        "WDM module should be using float64 precision, check the __init__ file."
    

def test_nu_d():
    r"""
    Test the normalised incomplete beta function.
    """
    d = 4

    x = np.linspace(0, 1, 1000)
    y = WDM.code.utils.Meyer.nu_d(x, d)

    assert y.shape == x.shape, \
        "nu_d should return an array of the same shape as input x"
    
    assert np.allclose(y[0], 0) and np.allclose(y[-1], 1), \
        "should have nu_d(0)=0 and nu_d(1)=1."
    
    x = np.array([-0.1, 1.1])
    y = WDM.code.utils.Meyer.nu_d(x, d)

    assert np.all(np.isnan(y)), \
        "function should return nan when x is outside [0,1]"
    

def test_Meyer():
    r"""
    Test the Meyer window function in WDM.
    """
    d = 4
    
    omega = np.linspace(-1, 1, 1000)
    Phi = WDM.code.utils.Meyer.Meyer(omega, d, A=0.25, B=0.5)

    assert omega.shape == Phi.shape, \
        "Meyer function should return an array of the same shape as input omega"
    
    integral = np.sum(Phi**2) * np.diff(omega)[0]

    assert np.isclose(integral, 1.0, atol=1e-3, rtol=1.0e-3), \
        "The integral of |Phi(om)|^2 w.r.t om should be 1"


def test_padding():
    r"""
    Test the padding function in the WDM class.
    """
    x = np.array([9.,9.,9.])

    N = 5

    for where in ['end', 'start', 'equal']:

        x_padded, mask = WDM.utils.pad_signal(x, N, where=where)

        assert len(x_padded) == N, \
            "Length of padded signal must be a multiple of Nf"
        
        assert jnp.all(jnp.array_equal(x, x_padded[mask])), \
            "Padded signal should match original signal when mask is applied"