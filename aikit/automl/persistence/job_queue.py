import os
import time
import redis
import persistqueue
import persistqueue.sqlqueue as pq
from persistqueue.sqlbase import with_conditional_transaction


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
        self.queue = pq.SQLiteQueue(path, db_file_name='queue.file')

    def enqueue(self, job):
        self.queue.put(job)

    def dequeue(self):
        row = self._safe_select().fetchone()
        if row and row[0] is not None:
            self.queue._delete(row[0])
            self.queue.total -= 1
            item = self.queue._serializer.loads(row[1])
            return item
        else:
            return None

    # Patch queue to avoid parallel select
    def _safe_select(self, *args):
        with self.queue.tran_lock:
            with self.queue._getter as tran:
                return tran.execute(self.queue._sql_select, args)

    def size(self):
        return self.queue._count()
