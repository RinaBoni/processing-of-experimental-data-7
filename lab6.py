class Cluster:
    def __init__(self):
        self.scores = []
        self.curX = 0
        self.curY = 0
        self.lastX = 0
        self.lastY = 0

    def size(self):
        return len(self.scores)

    def add(self, pt):
        self.scores.append(pt)

    def set_center(self):
        sum_x = sum(pt.x for pt in self.scores)
        sum_y = sum(pt.y for pt in self.scores)
        self.lastX = self.curX
        self.lastY = self.curY
        self.curX = sum_x / len(self.scores)
        self.curY = sum_y / len(self.scores)

    def clear(self):
        self.scores = []

    @staticmethod
    def initial_center(k, clusarr, vpt):
        size = len(vpt)
        step = size // k
        steper = 0
        for i in range(k):
            clusarr[i].curX = vpt[steper].x
            clusarr[i].curY = vpt[steper].y
            steper += step

    @staticmethod
    def bind(k, clusarr, vpt):
        for j in range(k):
            clusarr[j].clear()
        size = len(vpt)
        for i in range(size):
            min_distance = ((clusarr[0].curX - vpt[i].x) ** 2 + (clusarr[0].curY - vpt[i].y) ** 2) ** 0.5
            cl = clusarr[0]
            for j in range(1, k):
                distance = ((clusarr[j].curX - vpt[i].x) ** 2 + (clusarr[j].curY - vpt[i].y) ** 2) ** 0.5
                if min_distance > distance:
                    min_distance = distance
                    cl = clusarr[j]
            cl.add(vpt[i])
        return clusarr

    @staticmethod
    def start(k, clusarr, vpt):
        Cluster.initial_center(k, clusarr, vpt)
        while True:
            chk = 0
            Cluster.bind(k, clusarr, vpt)
            for j in range(k):
                clusarr[j].set_center()
                for p in range(k):
                    if clusarr[p].curX == clusarr[p].lastX and clusarr[p].curY == clusarr[p].lastY:
                        chk += 1
                if chk == k:
                    return