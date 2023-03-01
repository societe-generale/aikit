import os.path

import fsspec.implementations.local

from ._base import DataLoader, Format, check_write_type, EXTENSIONS, get_encoder
from .util import path_join


class FsDataLoader(DataLoader):

    def __init__(self, path: str):
        self.base_path = path
        self.filesystem = fsspec.implementations.local.LocalFileSystem(auto_mkdir=True)
        self.path_sep = os.path.pathsep

    def _get_filename(self, key, path, serialization_format):
        """ Returns the full filename for the given key and path. """
        if path is not None:
            return path_join(self.base_path, path, key + EXTENSIONS[serialization_format], pathsep=self.path_sep)
        else:
            return path_join(self.base_path, key + EXTENSIONS[serialization_format], pathsep=self.path_sep)

    def write(self, data, key, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        filename = self._get_filename(key, path, serialization_format)
        encoder = get_encoder(serialization_format)
        with self.filesystem.open(filename, "wb") as f:
            encoder.encode(data, f)

    def read(self, key, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        filename = self._get_filename(key, path, serialization_format)
        encoder = get_encoder(serialization_format)
        with self.filesystem.open(filename, "rb") as f:
            return encoder.decode(f)
