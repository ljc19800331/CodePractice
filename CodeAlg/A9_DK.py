
'''
Dijkstra Problem
1. motion planning -- tasks
2. A* algorithms -- tasks

'''

# My implementation
# ref: For Dijkstra, there is not need to maintain a best cost for each node since it's kind of greedy search.
# It always chooses the lowest cost node for next search. So the previous searched node always has a lower cost and has no chance to be updated.
# The first time we pop our destination from our queue, we have found the lowest cost to our destination.

'''
Algorithms: 
LeetCode 787 implementation
1. build the graph 
2. build the state: (cost, src, k) 
3. 

'''
from collections import defaultdict
import heapq
from heapq import *

# using heap is one of the most elegant way for this problem

n = 3                                               # Number of nodes
edges = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]     # The graph and the edge weights
src = 0             # source position
dst = 2             # distination
k = 1               # at most k stops -- in DK algorithm -- this k is not limited to all the cases -- pending

# pq: state
# g: the current dictionary
pq, g = [(0, src, k + 1)], defaultdict(dict)

# Define the graph
for s, d, w in edges:
    g[s][d] = w

while pq:

    # "heapq.heappop" always choose the minimum at the current level -- pending
    cost, s, k = heapq.heappop(pq)
    # cost: The current cost
    # s: The current node
    # k: The current k number of moves

    if s == dst:
        print("The final cost = ", cost)

    # ?
    if not k:
        continue

    for d in g[s]:
        heapq.heappush(pq, (cost + g[s][d], d, k - 1))

print("The final result is defined as ")

exit()

from collections import defaultdict
from heapq import *

def dijkstra(edges, f, t):

    # g is the graph that shows the nodes and weights
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    print("The g[l] = ", g)
    input("check the graph")

    # Define the values and the lists
    q, seen, mins = [(0, f, ())], set(), {f: 0}
    print("q = ", q)
    print("seen = ", seen)
    print("mins = ", mins)
    input("check the initialization")

    while q:

        (cost, v1, path) = heappop(q)
        print("The current path = ", path)

        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t:
                return (cost, path)

            # Find the neighbour nodes
            for c, v2 in g.get(v1, ()):

                # If this point was visited
                if v2 in seen:
                    continue

                # Update the closest path
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf")

if __name__ == "__main__":

    edges = [
        ("A", "B", 7),
        ("A", "D", 5),
        ("B", "C", 8),
        ("B", "D", 9),
        ("B", "E", 7),
        ("C", "E", 5),
        ("D", "E", 15),
        ("D", "F", 6),
        ("E", "F", 8),
        ("E", "G", 9),
        ("F", "G", 11)
            ]

    print("=== Dijkstra ===")
    print(edges)
    print("A -> E:")
    print(dijkstra(edges, "A", "E"))
    print("F -> G:")
    print(dijkstra(edges, "F", "G"))