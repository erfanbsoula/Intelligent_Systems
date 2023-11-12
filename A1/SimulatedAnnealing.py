import numpy as np

class SimulatedAnnealing():

    def __init__(self, energyFunc, tempScheduler=None, tempStart=10,
                 getNeighbor=None, neighborStd=1.0):

        self.energyFunc = energyFunc
        self.tempScheduler = tempScheduler
        self.tempStart = tempStart
        self.getNeighbor = getNeighbor
        self.neighborStd = neighborStd

    def get_neighbor(self, currentState):

        if self.getNeighbor == None:
            return np.random.normal(currentState, self.neighborStd)

        return self.getNeighbor(currentState)
    
    def get_temperature(self, r):

        if self.tempScheduler == None:
            return r * self.tempStart
        
        return self.tempScheduler(r)

    def is_accepted(self, e_curr, e_new, temp):

        if e_new < e_curr:
            return True

        return np.exp(-(e_new - e_curr) / temp) > np.random.random()

    def run(self, startPoint, k_max=1000):

        currentState = startPoint
        e_curr = self.energyFunc(currentState)

        for k in range(k_max):

            newState = self.get_neighbor(currentState)
            e_new = self.energyFunc(newState)
            temp = self.get_temperature(1 - k/k_max)

            if self.is_accepted(e_curr, e_new, temp):
                currentState = newState
                e_curr = e_new

        return currentState, e_curr
