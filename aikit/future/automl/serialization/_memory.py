from ._base import DataLoader, Format, check_write_type, get_encoder, get_io_class


class MemoryDataLoader(DataLoader):
    """
    Memory data loader.
    """

    def __init__(self):
        self._data = {}

    def write(self, data, key, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        composite_key = f"{path if path is not None else ''}-{key}-{serialization_format}"
        encoder = get_encoder(serialization_format)
        with get_io_class(serialization_format)() as f:
            encoder.encode(data, f)
            self._data[composite_key] = f.getvalue()

    def read(self, key, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        composite_key = f"{path if path is not None else ''}-{key}-{serialization_format}"
        encoder = get_encoder(serialization_format)
        with get_io_class(serialization_format)(self._data[composite_key]) as f:
            return encoder.decode(f)

    def read_from_cache(self, key, path=None, serialization_format=Format.PICKLE):
        return self.read(key, path, serialization_format)

    def exists(self, key, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        composite_key = f"{path if path is not None else ''}-{key}-{serialization_format}"
        return composite_key in self._data

    def get_all_keys(self, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        key_prefix = f"{path if path is not None else ''}-"
        key_postfix = f"-{serialization_format}"
        keys = []
        for key in self._data:
            if key.startswith(key_prefix) and key.endswith(key_postfix):
                keys += [key[len(key_prefix):-len(key_postfix)]]
        return keys
