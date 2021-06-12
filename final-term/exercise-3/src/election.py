import collections
import typing
import bisect

import networkx as nx

Candidate = collections.namedtuple("Candidate", ["id", "position"])


def _bisect_left(a: typing.List[Candidate], x: float, lo: int = 0, hi: int = None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid].position < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _position_distance(position1: float, position2: float) -> float:
    return abs(position1 - position2)


def nearest_candidate_id(voter_position: float, candidates: typing.List[Candidate]) -> int:
    # O(log c)
    greater_equal_index = _bisect_left(candidates, voter_position)

    if greater_equal_index == 0:
        return candidates[greater_equal_index].id

    if greater_equal_index == len(candidates):
        return candidates[greater_equal_index - 1].id

    if (_position_distance(voter_position, candidates[greater_equal_index - 1].position)
            <= _position_distance(voter_position, candidates[greater_equal_index].position)):
        return candidates[greater_equal_index - 1].id

    return candidates[greater_equal_index].id


def election(graph: nx.Graph, candidates: typing.List[Candidate]) -> typing.Dict[int, int]:
    """
    The graph contains, for each node, a property "peak_preference" which contains the id of the voted candidate.
    Returns a dict that maps ids of candidates to their total votes.
    """
    candidates = sorted(candidates, key=lambda candidate: candidate.position)
    votes = {candidate.id: 0 for candidate in candidates}
    for node in graph.nodes:
        voter_position = graph.nodes[node]['peak_preference']
        voted_candidate = nearest_candidate_id(voter_position, candidates)
        votes[voted_candidate] += 1

    return votes
