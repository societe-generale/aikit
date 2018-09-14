import pytest

from aikit.enums import DataTypes


@pytest.mark.parametrize("klass", [DataTypes])
def test_enums(klass):
    all_attrs = klass.__dict__
    assert "alls" in all_attrs
    for key, value in all_attrs.items():
        if isinstance(value, str) and key == value:
            if value not in all_attrs["alls"]:
                pytest.fail("{} should be in 'alls'".format(value))
