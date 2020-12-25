import numpy as np


class Point:
    def __init__(self, p: np.array):
        self.p = p
        self.prev: Point = None
        self.next: Point = None

    def act(self):
        # insert
        if self.prev.next is not self:
            self.prev.next = self
            self.next.prev = self
        # delete
        else:
            self.prev.next = self.next
            self.next.prev = self.prev


NIL = Point(None)


# < 0 if (p,q,r) turns clock wise
def turn(p, q, r) -> float:
    if p is NIL or q is NIL or r is NIL:
        return 1.0
    return (r.p[1]-p.p[1])*(q.p[0]-p.p[0]) - (q.p[1]-p.p[1])*(r.p[0]-p.p[0])


def time(p, q, r) -> float:
    if p is NIL or q is NIL or r is NIL:
        return np.inf
    return ((r.p[2]-p.p[2])*(q.p[0]-p.p[0]) - (q.p[2]-p.p[2])*(r.p[0]-p.p[0]))\
        / turn(p, q, r)


# input: a list of points in 3d
# output: indices of facets of lower hull
def hull(s: np.array):
    # attach original indices
    ind = np.arange(len(s)).reshape(len(s), 1)
    points = np.hstack((s, ind))

    # sort points in increasing x-coordinates
    points = points[np.argsort(points[:, 0])]

    # convert points into doubly linked list
    head = tail = Point(points[0])
    head.prev = NIL
    for p in points[1:]:
        temp = tail
        tail = Point(p)
        temp.next = tail
        tail.prev = temp
        tail.next = NIL

    E = hull_helper(head, len(s))

    # collect facets from events
    ch = []
    for e in E[:-1]:
        p = e.prev.p[3]
        q = e.p[3]
        r = e.next.p[3]
        ch.append([p, q, r])
        e.act()

    return ch


def hull_helper(head: Point, n: int):
    # base case
    if n == 1:
        head.prev = head.next = NIL
        return [NIL]

    u = head
    i = 0
    while i < int(n/2) - 1:
        u = u.next
        i += 1
    mid = v = u.next
    # recurse on left and right sides
    E1 = hull_helper(head, int(n/2))
    E2 = hull_helper(mid, n - int(n/2))
    E = []

    # find initial bridge
    while True:
        if turn(u, v, v.next) < 0:
            v = v.next
        elif turn(u.prev, u, v) < 0:
            u = u.prev
        else:
            break

    oldt = -np.inf
    i = j = 0
    while True:
        t = [time(E1[i].prev, E1[i], E1[i].next),
             time(E2[j].prev, E2[j], E2[j].next),
             time(u, u.next, v),
             time(u.prev, u, v),
             time(u, v.prev, v),
             time(u, v, v.next)]

        # find next event
        newt = np.inf
        k = 0
        while k < 6:
            if t[k] > oldt and t[k] < newt:
                mink = k
                newt = t[k]
            k += 1

        # break if no next event
        if newt == np.inf:
            break
        if mink == 0:
            if E1[i].p[0] < u.p[0]:  # L undergoes an event to the left of u
                E.append(E1[i])
            E1[i].act()
            i += 1
        elif mink == 1:
            if E2[j].p[0] > v.p[0]:  # R undergoes an event to the right of v
                E.append(E2[j])
            E2[j].act()
            j += 1
        elif mink == 2:              # (u,u+,v) turns counterclockwise
            u = u.next
            E.append(u)
        elif mink == 3:              # (u-,u,v) turns clockwise
            E.append(u)
            u = u.prev
        elif mink == 4:              # (u,v-,v) turns counterclockwise
            v = v.prev
            E.append(v)
        else:                        # (u,v,v+) turns clockwise
            E.append(v)
            v = v.next
        oldt = newt
    u.next = v
    v.prev = u

    # revert to initial hull
    for e in reversed(E):
        if e.p[0] <= u.p[0] or e.p[0] >= v.p[0]:
            e.act()
            if e == u:
                u = u.prev
            elif e == v:
                v = v.next
        else:
            u.next = e
            e.prev = u
            v.prev = e
            e.next = v
            if e.p[0] < mid.p[0]:
                u = e
            else:
                v = e
    E.append(NIL)

    return E
