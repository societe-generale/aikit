import hashlib
import json
from io import StringIO

from .json import SpecialJSONEncoder


def md5_hash(ob):
    """
    Hash of an object, constant across computers/process.
    Works with non hashable python object.
    """
    s = StringIO()
    json.dump(ob, s, cls=SpecialJSONEncoder)
    m = hashlib.md5()
    m.update(s.getvalue().encode("utf-8"))
    return m.hexdigest()
