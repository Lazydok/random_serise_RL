# 4 x 4 x 1
# sec window * etf * close price
# etf = [reverage, revarage2, reverse, random]
import numpy as np

class GenRL1:
    def __init__(self, size=60 * 60 * 3):
        self.__makeData__(size)
        self.windowSize = 8
        self.idx = self.windowSize
        self.terminate = False
        self.len = len(self.etf1)

    def __len__(self):
        return self.len

    def __makeData__(self, iterMax):


        dCnt = 0
        d = 20 + int(70*np.random.rand())

        val = 0.5
        val2 = 0.5
        call = 1

        etf1 = []
        etf2 = []
        etf3 = []
        etf4 = []

        for i in range(iterMax):
            dCnt += 1
            if d == dCnt:
                call = -1 * call
                dCnt = 0
                d = 20 + int(70 * np.random.rand())
            # print(np.random.rand())
            val += call * (1 / iterMax) * np.random.rand()
            val2 += 1 / iterMax * (np.random.rand() - 0.5)
            etf1.append(val)
            etf2.append(val + (1/iterMax) * (np.random.rand() - 0.5))
            etf3.append(1 - (val + (1/iterMax) * (np.random.rand() - 0.5)))
            etf4.append(val2)

        self.etf1 = etf1[:-2]
        self.etf2 = etf2[2:]
        self.etf3 = etf3[2:]
        self.etf4 = etf4[2:]

    def next(self):
        if self.idx == (self.len - 1):
            self.terminate = True

        state = np.array([self.etf1[self.idx - self.windowSize:self.idx],
                 self.etf2[self.idx - self.windowSize:self.idx],
                 self.etf3[self.idx - self.windowSize:self.idx],
                 self.etf4[self.idx - self.windowSize:self.idx]
                ])

        if not self.terminate:
            self.idx += 1
            return state.reshape((4, self.windowSize, -1)), self.terminate
        else:
            return None, self.terminate