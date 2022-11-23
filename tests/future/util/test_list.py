from aikit.future.util.list import unlist, tuple_include, unnest_tuple, intersect, diff


def test_unlist():
    assert unlist([[1, 10], [32]]) == [1, 10, 32]
    assert unlist([[10], [11], [], [45]]) == [10, 11, 45]


def test_tuple_include():
    assert tuple_include((1, 2), (1, 2, 3))
    assert not tuple_include((1, 2, 3), (1, 2))
    assert tuple_include((1, 2, 3), (1, 2, 3))
    assert not tuple_include((1, 2, 4), (1, 2, 3))


def test_unnest_tuple():
    examples = [
        ((1, 2, 3), (1, 2, 3)),
        ((1, (2, 3)), (1, 2, 3)),
        ((1, (2, (3,))), (1, 2, 3)),
        (((1,), (2,), (3, 4)), (1, 2, 3, 4)),
    ]

    for nested_tuple, unnested_tuple in examples:
        assert unnest_tuple(nested_tuple) == unnested_tuple


def test_diff():
    list1 = [1, 2, 3]
    list2 = [3, 4, 5]

    assert diff(list1, list2) == [1, 2]
    assert diff(list2, list1) == [4, 5]

    assert diff(list1, []) == list1

    list1 = ["a", "b", "c"]
    list2 = ["d", "c", "e"]

    assert diff(list1, list2) == ["a", "b"]
    assert diff(list2, list1) == ["d", "e"]

    assert diff(list1, []) == list1

    assert isinstance(diff((1, 2, 3), (1, 2)), tuple)


def test_intersect():
    list1 = [1, 2, 3]
    list2 = [3, 4, 5]

    assert intersect(list1, list2) == [3]
    assert intersect(list2, list1) == [3]

    list1 = [1, 2, 3, 4]
    list2 = [4, 3, 5, 6]

    assert intersect(list1, list2) == [3, 4]
    assert intersect(list2, list1) == [4, 3]

    assert intersect(list1, []) == []

    list1 = ["a", "b", "c"]
    list2 = ["d", "c", "e"]

    assert intersect(list1, list2) == ["c"]
    assert intersect(list2, list1) == ["c"]

    assert intersect(list1, []) == []
    assert isinstance(diff((1, 2, 3), (1, 2)), tuple)
