from aikit.future.util.list import unlist


def test_unlist():
    assert unlist([[1, 10], [32]]) == [1, 10, 32]
    assert unlist([[10], [11], [], [45]]) == [10, 11, 45]
