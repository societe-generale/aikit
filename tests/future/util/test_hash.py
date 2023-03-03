from aikit.future.util.hash import md5_hash


def test_md5_hash():
    ob1 = ((1, 2), {"1": "2"})
    ob2 = ((1, 2), {1: "2"})
    ob3 = ([1, 2], {1: 2})

    h1 = md5_hash(ob1)
    h2 = md5_hash(ob2)
    h3 = md5_hash(ob3)
    assert len({h1, h2, h3}) == 3  # Test 3 different hash

    assert h1 == "d5f3de055dd4049def2766a9a6a3e914"
    assert h2 == "e8c67e026e91872ef85bc56cf67ab97a"
    assert h3 == "f409ec84efccad047568fa1ca5d0f990"
