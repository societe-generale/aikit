import numpy as np

from aikit.future.util.num import is_number


def test__is_number():
    examples_numbers = [10, 10.1, np.float16(19.0), np.int32(12), np.nan]

    for x in examples_numbers:
        assert is_number(x)

    examples_not_numbers = ["toto", None, "a", "10.0", "0"]
    for x in examples_not_numbers:
        assert not is_number(x)
