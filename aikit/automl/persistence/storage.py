import os
import json
import pickle
import pandas as pd
from smart_open import open
from aikit.tools.json_helper import SpecialJSONEncoder, SpecialJSONDecoder



class Storage:

    def __init__(self, path):
        self.path = path
        self.backend = FileStorage(path)

    def full_path(self, key, folder=None):
        if folder is not None and not self.exists(folder):
            self.mkdir(folder)
        return self.backend.full_path(key, folder)

    def save(self, data, key, folder=None):
        with open(self.full_path(key + '.txt', folder), 'w') as file:
            file.write(data)

    def load(self, key, folder=None):
        with open(self.full_path(key + '.txt', folder), 'r') as file:
            return file.read()

    def save_pickle(self, data, key, folder=None):
        with open(self.full_path(key + '.pkl', folder), 'wb') as file:
            pickle.dump(data, file)

    def load_pickle(self, key, folder=None):
        with open(self.full_path(key + '.pkl', folder), 'rb') as file:
            return pickle.load(file)

    def save_json(self, data, key, folder=None):
        with open(self.full_path(key + '.json', folder), 'w') as file:
            json.dump(data, file, indent=4)

    def load_json(self, key, folder=None):
        with open(self.full_path(key + '.json', folder), 'r') as file:
            return json.load(file)

    def save_special_json(self, data, key, folder=None):
        with open(self.full_path(key + '.json', folder), 'w') as file:
            json.dump(data, file, cls=SpecialJSONEncoder, indent=4)

    def load_special_json(self, key, folder=None):
        with open(self.full_path(key + '.json', folder), 'r') as file:
            return json.load(file, cls=SpecialJSONDecoder)

    def save_csv(self, data, key, folder=None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Only pandas DataFrame can be saved to csv')
        with open(self.full_path(key + '.csv', folder), 'w') as file:
            data.to_csv(file, index=False)

    def load_csv(self, key, folder=None):
        with open(self.full_path(key + '.csv', folder), 'r') as file:
            return pd.read_csv(file)

    def exists(self, key, folder=None):
        return self.backend.exists(self.full_path(key, folder))

    def listdir(self, folder):
        if not self.exists(folder):
            self.mkdir(folder)
        return self.backend.listdir(folder)

    def mkdir(self, folder):
        self.backend.mkdir(folder)

    def remove(self, key, folder=None):
        self.backend.remove(self.full_path(key, folder))


class FileStorage:

    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def full_path(self, key, folder=None):
        if folder is None:
            return os.path.join(self.path, key)
        else:
            return os.path.join(self.path, folder, key)

    def exists(self, path):
        return os.path.exists(os.path.join(self.path, path))

    def listdir(self, folder):
        return os.listdir(os.path.join(self.path, folder))

    def mkdir(self, folder):
        os.mkdir(os.path.join(self.path, folder))

    def remove(self, path):
        try:
            os.remove(os.path.join(self.path, path))
        except:
            pass
