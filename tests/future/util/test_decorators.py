from aikit.future.util.decorators import enforce_init, singleton


@enforce_init
class A:
    def __init__(self):
        self.arg = "arg"
        self._protected = "protected"
        self.__private = "private"

    def set_other_arg(self):
        self.b = "other_arg"  # noqa


def test_enforce_init():
    a = A()
    a.arg = "myArg"
    a._protected = "protected"
    try:
        a.set_other_arg()
        assert False
    except TypeError:
        pass


def test_singleton():
    @singleton
    class Foo(object):
        def __init__(self, f=1):
            self.f = f

    f1 = Foo()
    f2 = Foo()

    assert f1 is f2
    f1.f = 10
    assert f2.f == 10
