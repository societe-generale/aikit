import json
import pickle

import numpy as np
import pandas as pd

from ._base import Encoder, register_encoder, Format
from ...util.json import SpecialJSONEncoder, SpecialJSONDecoder


class TxtEncoder(Encoder):

    def encode(self, data, f):
        f.write(data)

    def decode(self, f):
        return f.read()


class JsonEncoder(Encoder):

    def encode(self, data, f):
        json.dump(data, f, cls=SpecialJSONEncoder)

    def decode(self, f):
        return json.load(f, cls=SpecialJSONDecoder)


class PickleEncoder(Encoder):

    def encode(self, data, f):
        pickle.dump(data, f)

    def decode(self, f):
        return pickle.load(f)


class CsvEncoder(Encoder):

    def encode(self, data, f):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a numpy.ndarray or a pandas.DataFrame in order to save as CSV, "
                            "got {type(data)}")
        data.to_csv(f, sep=";", encoding="utf-8", index=False)

    def decode(self, f):
        return pd.read_csv(f, sep=";", encoding="utf-8", index_col=None)


register_encoder(Format.TEXT, TxtEncoder())
register_encoder(Format.JSON, JsonEncoder())
register_encoder(Format.PICKLE, PickleEncoder())
register_encoder(Format.CSV, CsvEncoder())
