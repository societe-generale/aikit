import logging
import os.path

import cachetools.func
import fsspec.implementations.local

from ._base import DataLoader, Format, check_write_type, EXTENSIONS, get_encoder, get_write_mode, get_read_mode
from .util import path_join, path_split

_logger = logging.getLogger(__name__)


class FsDataLoader(DataLoader):

    def __init__(self, path: str, session_id: str):
        # TODO path_sep should depend on backend
        self.path_sep = os.path.sep
        self.base_path = path_join(path, session_id, pathsep=self.path_sep)
        _logger.info(f"Using FsDataLoader with base path: {self.base_path}")
        # TODO switch implementation based on protocol
        self.filesystem = fsspec.implementations.local.LocalFileSystem(auto_mkdir=True)
        self.filesystem.mkdirs(self.base_path, exist_ok=True)

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
        with self.filesystem.open(filename, get_write_mode(serialization_format)) as f:
            encoder.encode(data, f)

    def read(self, key, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        filename = self._get_filename(key, path, serialization_format)
        encoder = get_encoder(serialization_format)
        with self.filesystem.open(filename, get_read_mode(serialization_format)) as f:
            return encoder.decode(f)

    def exists(self, key, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        filename = self._get_filename(key, path, serialization_format)
        return self.filesystem.exists(filename)

    @cachetools.func.lru_cache
    def read_from_cache(self, key, path=None, serialization_format=Format.PICKLE):
        return self.read(key, path, serialization_format)

    def get_all_keys(self, path=None, serialization_format=Format.PICKLE):
        check_write_type(serialization_format)
        all_files = self.filesystem.glob(self._get_filename("*", path, serialization_format))
        all_keys = [os.path.splitext(path_split(f)[1])[0] for f in all_files]
        return all_keys
