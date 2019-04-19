# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:18:52 2018

@author: Lionel Massoulard
"""

import pandas as pd
import numpy as np

import os.path
import os

from lockfile import LockFile
import pickle
import time

from glob import glob

from aikit.tools.json_helper import save_json, load_json

# In[]


class SavingType(object):
    json = "json"
    pickle = "pickle"
    csv = "csv"
    txt = "txt"

    alls = (json, pickle, csv, txt)


# class SharedDico(object):
#    def __init__(self,
#                 path,
#                 data_persister):
#
#        self.path = path
#        self.data_persister = data_persister
#
#    def __getitem__(self,key):
#        if self.data_persister.exist(key = key, path = self.path,write_type = SavingType.json):
#            return self.data_persister.read(key = key, path = self.path,write_type = SavingType.pickle)
#        else:
#            raise KeyError(key)
#
#    def __setitem__(self,key,item):
#        self.data_persister.write(data = item,key = key, path = self.path, write_type = SavingType.pickle)


class SharedInteger(object):
    def __init__(self, data_persister, path, key, value_if_dont_exist=0):
        self.data_persister = data_persister
        self.path = path
        self.key = key

        if not self.data_persister.exists(key=self.key, path=self.path, write_type=SavingType.json):
            self.value = value_if_dont_exist

    @property
    def value(self):
        res = self.data_persister.read(path=self.path, key=self.key, write_type=SavingType.json, _dont_lock=False)
        return res

    @value.setter
    def value(self, new_value):
        self.data_persister.write(
            data=new_value, path=self.path, key=self.key, write_type=SavingType.json, _dont_lock=False
        )

    def _modify(self, f):

        with self.data_persister.get_lock(path=self.path, key=self.key, write_type=SavingType.json):

            current_value = self.data_persister.read(
                path=self.path, key=self.key, write_type=SavingType.json, _dont_lock=True
            )  # already lock
            new_value = f(current_value)
            self.data_persister.write(
                data=new_value, path=self.path, key=self.key, write_type=SavingType.json, _dont_lock=True
            )

        return new_value

    def inc(self):
        return self._modify(lambda x: x + 1)

    def dec(self):
        return self._modify(lambda x: x - 1)

        # with self.data_persister.get_lock(path = self.path, key = self.key, write_type = SavingType.json):


class DummyDataPerister(object):
    def __init__(self, base_folder):

        self.base_folder = base_folder
        self._data = {}

    @classmethod
    def get_write_type(cls, write_type):

        if write_type is None:
            raise TypeError("write_type shouldn't be None")

        if not isinstance(write_type, str):
            raise TypeError("write_type should be a string, not a %s" % type(write_type))

        write_type = write_type.lower()
        if isinstance(write_type, str) and len(write_type) > 0 and write_type[0] == ".":
            write_type = write_type[1:]

        if write_type not in SavingType.alls:
            raise ValueError("I don't know how to handle that type of data : %s" % write_type)

        return write_type

    def write(self, data, key, path=None, write_type=SavingType.pickle):
        complete_path = (path, key, self.get_write_type(write_type))

        self._data[complete_path] = data

    def read(self, key, path=None, write_type=SavingType.pickle):
        complete_path = (path, key, self.get_write_type(write_type))

        return self._data[complete_path]

    read_from_cache = read

    def exists(self, key, path=None, write_type=SavingType.pickle):
        complete_path = (path, key, self.get_write_type(write_type))
        return complete_path in self._data

    def delete(self, key, path=None, write_type=SavingType.pickle):
        complete_path = (path, key, self.get_write_type(write_type))
        del self._data[complete_path]

    def alls(self, path, write_type=SavingType.pickle):
        write_type = self.get_write_type(write_type)
        return [key for (p, key, w) in self._data.keys() if (p, w) == (path, write_type)]


class Queue(object):
    """ class representing a queue """

    def __init__(self, data_persistor, path, write_type=SavingType.json, max_queue_size=None, random=False):
        self.data_persistor = data_persistor
        self.path = path
        self.write_type = write_type
        self.max_queue_size = max_queue_size
        self.random = random

    def remove(self):
        data = self.data_persistor.remove_from_queue(path=self.path, write_type=self.write_type, random=self.random)
        return data

    def add(self, data):
        res = self.data_persistor.add_in_queue(
            data=data, path=self.path, write_type=self.write_type, max_queue_size=self.max_queue_size
        )
        return res


class _DummyLock(object):
    """ object that can work using 'with' but that actually don't do anything """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


class FolderDataPersister(object):
    """ object to persiste data in a given folder
    each data will be saved in a given path, with a given key, and a given type

    object can:
        * read  a data given its  key/path/type
        * write a data in a given key/path/type
        * tell if a data exists
        * delete a given data
        * tells everything in a given path

    """

    def __init__(self, base_folder):
        self.base_folder = base_folder

        if not os.path.isdir(self.base_folder):
            with LockFile("_".join(os.path.split(self.base_folder))):
                os.makedirs(self.base_folder)

        self._cache = {}

    def get_complete_path(self, key, path, write_type):

        key = str(key)

        if path is None:
            complete_path = os.path.join(self.base_folder, key) + "." + write_type
        else:
            complete_path = os.path.join(self.base_folder, path, key) + "." + write_type

        folder = os.path.split(complete_path)[0]
        if not os.path.isdir(folder):
            with LockFile("_".join(os.path.split(folder))):
                os.makedirs(folder)

        return complete_path

    @classmethod
    def get_write_type(cls, write_type):

        if write_type is None:
            raise TypeError("write_type shouldn't be None")

        if not isinstance(write_type, str):
            raise TypeError("write_type should be a string, not a %s" % type(write_type))

        write_type = write_type.lower()
        if isinstance(write_type, str) and len(write_type) > 0 and write_type[0] == ".":
            write_type = write_type[1:]

        if write_type not in SavingType.alls:
            raise ValueError("I don't know how to handle that type of data : %s" % write_type)

        return write_type

    def get_lock(self, path, key, write_type):
        write_type = self.get_write_type(write_type)
        complete_path = self.get_complete_path(key, path, write_type)
        lock_file_key = self.get_lock_file(complete_path)
        return LockFile(lock_file_key)

    def write(self, data, key, path=None, write_type=SavingType.pickle, _dont_lock=False):
        """ write a given key """
        write_type = self.get_write_type(write_type)

        complete_path = self.get_complete_path(key, path, write_type)

        lock_file_key = self.get_lock_file(complete_path)

        if _dont_lock:
            lock = _DummyLock()
            # In that case the '_DummyLock' doesn't do anything so I wont lock
            # No locking can be usefull in the case where the lock append outside the function
        else:
            lock = LockFile(lock_file_key)

        #################
        ### JSON type ###
        #################
        if write_type == SavingType.json:

            with lock:
                save_json(data, complete_path)

        ################
        ### CSV type ###
        ################
        elif write_type == SavingType.csv:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            elif not isinstance(data, pd.DataFrame):
                raise TypeError("I don't know how to save this type %s to csv" % type(data))

            with lock:
                data.to_csv(complete_path, sep=";", encoding="utf-8", index=False)

        ###################
        ### PICKLE type ###
        ###################
        elif write_type == SavingType.pickle:

            with lock:
                with open(complete_path, "wb") as f:
                    pickle.dump(data, f)

        ################
        ### TXT type ###
        ################
        elif write_type == SavingType.txt:

            with lock:
                with open(complete_path, "w", encoding="utf-8") as f:
                    f.write(data)
        else:
            raise ValueError("Unknown writting type %s" % write_type)

    # def get_lock(key, path, write_type)

    def read_from_cache(self, key, path=None, write_type=SavingType.pickle, _dont_lock=False):
        """ read a given key from the cache, or load it and store in cache if isn't present

        Remark : do not use with object that can change
        """
        dico_key = (key, path, write_type)

        try:
            result = self._cache[dico_key]  #
            # Try to find it in the cache...
        except KeyError:
            # ... if not present

            result = self.read(key=key, path=path, write_type=write_type, _dont_lock=_dont_lock)
            # Read it ...

            self._cache[dico_key] = result
            # and store it for next time

        return result

    def read(self, key, path=None, write_type=SavingType.pickle, _dont_lock=False):
        """ read a given key """

        write_type = self.get_write_type(write_type)
        complete_path = self.get_complete_path(key, path, write_type)

        lock_file_key = self.get_lock_file(complete_path)

        if _dont_lock:
            lock = _DummyLock()
        else:
            lock = LockFile(lock_file_key)

        with lock:
            if not os.path.exists(complete_path):
                raise ValueError("The key %s doesn't exist in %s" % (key, path))

        if write_type == SavingType.json:

            with lock:
                data = load_json(complete_path)

        elif write_type == SavingType.pickle:

            with lock:
                with open(complete_path, "rb") as f:
                    data = pickle.load(f)

        elif write_type == SavingType.csv:
            with lock:
                data = pd.read_csv(complete_path, sep=";", encoding="utf-8")

        elif write_type == SavingType.txt:
            with lock:
                with open(complete_path, "r", encoding="utf-8") as f:
                    data = f.read()

        else:
            raise ValueError("Unknown writting type %s" % write_type)

        return data

    def exists(self, key, path=None, write_type=SavingType.pickle):
        """ does a key exists """

        write_type = self.get_write_type(write_type)
        complete_path = self.get_complete_path(key, path, write_type)

        lock_file_key = self.get_lock_file(complete_path)

        with LockFile(lock_file_key):
            result = os.path.exists(complete_path)

        return result

    def delete(self, key, path=None, write_type=SavingType.pickle):
        """ delete a key """

        write_type = self.get_write_type(write_type)
        complete_path = self.get_complete_path(key, path, write_type)

        lock_file_key = self.get_lock_file(complete_path)

        with LockFile(lock_file_key):
            os.remove(complete_path)

    def alls(self, path=None, write_type=SavingType.pickle):
        """ all keys within a path """

        write_type = self.get_write_type(write_type)
        complete_path = self.get_complete_path(key="*", path=path, write_type=write_type)

        t0 = time.time()
        max_weight = 10
        while True:
            all_locks = glob(complete_path + "_filelock.lock")
            if len(all_locks) == 0:
                break

            print("within 'alls' : lock file present... I'll wait")
            time.sleep(1)
            t1 = time.time()
            if t1 - t0 >= max_weight:
                break

        all_files = glob(complete_path)

        all_keys = [os.path.splitext(os.path.split(f)[1])[0] for f in all_files]
        return all_keys

    ######################
    ### Shared Integer ###
    ######################
    def new_shared_integer(self, path, key):
        return SharedInteger(data_persister=self, path=path, key=key)

    #############
    ### Queue ###
    #############
    def new_queue(self, path, write_type=SavingType.json, max_queue_size=None, random=False):
        """ create a new queue """
        return Queue(
            data_persistor=self, path=path, write_type=write_type, max_queue_size=max_queue_size, random=random
        )

    def add_in_queue(self, data, path, write_type=SavingType.pickle, max_queue_size=None):
        complete_path = self.get_complete_path(key="queue", path=path, write_type=write_type)
        lock_file_folder = self.get_lock_file(complete_path, is_folder=True)

        with LockFile(lock_file_folder):

            all_items = self.alls(path, write_type=write_type)
            if max_queue_size is not None and len(all_items) >= max_queue_size:
                return False

            if len(all_items) > 0:
                key = max([int(i) for i in all_items]) + 1
            else:
                key = 0

            self.write(data=data, key=key, path=path, write_type=write_type)

        return True

    def remove_from_queue(self, path, write_type=SavingType.pickle, random=False):

        complete_path = self.get_complete_path(key="queue", path=path, write_type=write_type)
        lock_file_folder = self.get_lock_file(complete_path, is_folder=True)

        with LockFile(lock_file_folder):

            all_items = sorted(self.alls(path, write_type=write_type), key=lambda x: int(x))
            if len(all_items) == 0:
                data = None
            else:
                if random:
                    choice = np.random.choice(list(range(len(all_items))), 1)[0]
                else:
                    choice = 0

                key = all_items[choice]
                data = self.read(key=key, path=path, write_type=write_type)
                self.delete(key=key, path=path, write_type=write_type)

        return data

    @classmethod
    def get_lock_file(cls, complete_path, is_folder=False):

        if is_folder:
            lock_file_key = complete_path + "_folderlock"
        else:
            lock_file_key = complete_path + "_filelock"

        return lock_file_key


# In[]
