import os
import time
import redis
import persistqueue
import persistqueue.sqlqueue as pq


class JobsQueue:
    # Queue implementation for redis
    # https://gist.github.com/lewellent/d5b471bfd677c7121244

    def __init__(self, path):
        self.path = path
        self.backend = FileQueue(path)

    def enqueue(self, job):
        self.backend.enqueue(job)

    def dequeue(self):
        return self.backend.dequeue()

    def size(self):
        return self.backend.size()


class FileQueue:

    def __init__(self, path):
        path = os.path.join(path, 'queue')
        self.queue = pq.SQLiteQueue(path, auto_commit=True, db_file_name='queue')

    def enqueue(self, job):
        self.queue.put(job)

    def dequeue(self):
        try:
            return self.queue.get(timeout=1)
        except persistqueue.exceptions.Empty:
            return None

    def size(self):
        return self.queue._count()
