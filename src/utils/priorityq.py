import itertools
from heapq import heappush, heappop

REMOVED = '<removed-task>'  # placeholder for a removed task


class PriorityQueue:

    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = itertools.count()  # unique sequence count

    def add(self, task, priority=0):  # O(logN)
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove(self, task):  # O(1)
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = REMOVED
        return entry[0]

    def pop(self):  # O(logN)
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def top(self):  # O(1)
        if len(self.pq) != 0:
            return self.pq[0]
        else:
            raise KeyError('top from an empty priority queue')

    def __len__(self):
        return len(self.pq)
