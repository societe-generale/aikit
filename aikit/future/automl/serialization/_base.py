import abc
import io


class Format(object):
    JSON = "json"
    PICKLE = "pickle"
    CSV = "csv"
    TEXT = "txt"
    ALL = (JSON, PICKLE, CSV, TEXT)


EXTENSIONS = {
    Format.JSON: ".json",
    Format.PICKLE: ".pkl",
    Format.CSV: ".csv",
    Format.TEXT: ".txt"
}


def check_write_type(fmt: str):
    """ Check if the specified format is valid.

    Parameters
    ----------
    fmt: str
        The output format to check
    """

    if fmt is None:
        raise TypeError("write_type must not be None")

    if not isinstance(fmt, str):
        raise TypeError(f"write_type should be a string, not a {type(fmt)}")

    fmt = fmt.lower()
    if fmt not in Format.ALL:
        raise ValueError(f"write_type must be one of {Format.ALL}, got {fmt}")

    return fmt


ENCODERS = {}


def get_write_mode(fmt: Format):
    """ Get the write mode for the given format. """
    if fmt in (Format.TEXT, Format.CSV, Format.JSON):
        return "w"
    else:
        return "wb"


def get_read_mode(fmt: Format):
    """ Get the read mode for the given format. """
    if fmt in (Format.TEXT, Format.CSV, Format.JSON):
        return "r"
    else:
        return "rb"


def get_io_class(fmt: Format):
    """ Get the IO object for the given format. """
    if fmt in (Format.TEXT, Format.CSV, Format.JSON):
        return io.StringIO
    else:
        return io.BytesIO


def get_encoder(fmt: Format):
    """ Get the encoder for the given format. """
    if fmt not in ENCODERS:
        raise ValueError(f"Unknown format: {fmt}")
    return ENCODERS[fmt]


def register_encoder(fmt: Format, encoder):
    ENCODERS[fmt] = encoder


class DataLoader(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def write(self, data, key, path=None, serialization_format=Format.PICKLE):
        raise NotImplementedError()

    @abc.abstractmethod
    def read(self, key, path=None, serialization_format=Format.PICKLE):
        raise NotImplementedError()

    @abc.abstractmethod
    def read_from_cache(self, key, path=None, serialization_format=Format.PICKLE):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_all_keys(self, path=None, serialization_format=Format.PICKLE):
        raise NotImplementedError()


class Encoder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def encode(self, data, f):
        pass

    @abc.abstractmethod
    def decode(self, f):
        pass
