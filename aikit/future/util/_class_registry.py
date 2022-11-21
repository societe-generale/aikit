class ClassRegistry:
    """ simple class to handle dictionary of class name -> type """

    def __init__(self):
        self._mapping = {}

    def add_klass(self, klass):
        self._mapping[klass.__name__] = klass

    def __getitem__(self, klass_name):
        return self._mapping[klass_name]

    def __repr__(self):
        result = ["registered classes: "]\
                 + [s for s in sorted(self._mapping.keys())]
        return "\n".join(result)

    def get(self, key, default=None):
        return self._mapping.get(key, default)


CLASS_REGISTRY = ClassRegistry()
