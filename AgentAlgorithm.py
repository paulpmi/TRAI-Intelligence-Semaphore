import random
from xml.dom import minidom
import numpy as np

import traci
import numpy

from copy import deepcopy


import keras
from keras import Input
from keras.engine import Model
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop

class Individual:
    def __init__(self, nHidden, nrHidden):

        self.selectedEpisode = "D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/cluj-centru-500/osm.net.xml"
        self.selectedBackup = "D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/cluj-centru-500/backup.xml"
        self.selectedStartup = "D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/cluj-centru-500/osm.sumocfg"

        self.boot()

        self.semaphorePhases = []
        self.lanesIds = []
        self.sumHalting = []
        self.id = []
        self.originals = []

        self.haltingDelimiter = []

        xmldoc = minidom.parse(self.selectedEpisode, )
        itemlist = xmldoc.getElementsByTagName('tlLogic')


        laneI = 0
        for item in itemlist:
            self.id.append(item.attributes['id'].value)
            self.lanesIds.append(traci.trafficlights.getControlledLanes(self.id[laneI]))
            self.haltingDelimiter.append(len(self.lanesIds[laneI]))
            self.sumHalting += [0 for _ in range(len(self.lanesIds[laneI]))]

            laneI += 1
            power = item.getElementsByTagName('phase')
            phase = []
            for p in power:
                phase.append(int(p.attributes['duration'].value))
                nr = 0
                for s in p.attributes['state'].value:
                    if s is 'g' or s is 'G':
                        nr += 1
            self.ids = phase
            self.semaphorePhases += phase
            self.originals.append(phase)

        self.starter = [False for _ in range(len(self.lanesIds))]

        nInput = len(self.semaphorePhases) + len(self.sumHalting)

        self.neuronsInput = 2
        self.neuronsOutput = len(self.semaphorePhases)
        self.neuronsHidden = 5
        self.nrHidden = 5

        self.start = False

        self.fit = 0

        self.inputLayer = []
        self.hiddenLayers = []
        self.outputLayer = []

        threshold = 2
        thresholdSyntesiser = 1
        self.inputLayer = threshold * numpy.random.random((self.neuronsInput, self.neuronsHidden)) - thresholdSyntesiser
        for i in range(self.nrHidden - 1):
            weights = threshold * numpy.random.random((self.neuronsHidden, self.neuronsHidden)) - thresholdSyntesiser
            self.hiddenLayers.append(weights)

        self.outputLayer = threshold * numpy.random.random((self.neuronsHidden, self.neuronsOutput)) - thresholdSyntesiser

        self.inputLayer = numpy.asarray(self.inputLayer)
        self.outputLayer = numpy.asarray(self.outputLayer)

        self.pastIterationPhases = self.semaphorePhases

        #traci.close(wait=False)

    def tahn(self, s):
        return 1 / (1 + numpy.exp(-s))

    def activate(self, nodeVal, weights):
        return numpy.dot(nodeVal, weights)

    def predict(self, vehicleHalted, previousPhases):
        dataVehicles = []
        delimiter = 0
        for i in self.haltingDelimiter:
            dataVehicles.append(sum(vehicleHalted[delimiter:delimiter+i]))
            delimiter += i

        dataPhases = []
        delimiter = 0
        for i in self.originals:
            dataPhases.append(sum(previousPhases[delimiter:delimiter+len(i)]))
            delimiter += len(i)

        data = []
        for i in range(len(dataPhases)):
            data.append([dataVehicles[i], dataPhases[i]])

        data = numpy.asarray(data, numpy.float64)
        #data = data.reshape(data.size, 2)
        inputLayerOutput = self.tahn(self.activate(data, self.inputLayer))

        hiddenLayerOutput = []
        for i in range(self.nrHidden - 1):
            hiddenLayerOutput = self.tahn(self.activate(inputLayerOutput, self.hiddenLayers[i]))
            inputLayerOutput = deepcopy(hiddenLayerOutput)

        output = numpy.multiply(self.tahn(self.activate(hiddenLayerOutput, self.outputLayer)), 100)
        return output

    def modifyXML(self, newState):

        #newState = newState.tolist()
        xmldoc1 = minidom.parse(self.selectedEpisode)

        a = 0
        for s in xmldoc1.getElementsByTagName('tlLogic'):

            for id in self.id:
                if s.attributes['id'].value == id:

                    q = s.getElementsByTagName('phase')

                    k = 0
                    for phase in q:
                        phase.attributes['duration'].value = str(newState[a][k])
                        k += 1
                    a += 1


        xmldoc1.writexml(
            open(self.selectedEpisode, 'w', encoding="utf-8"))


        k = 0
        xmldoc1 = minidom.parse("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/rilsa1_tls.add.xml")

        for s in xmldoc1.getElementsByTagName('phase'):
            s.attributes['duration'].value = str(newState[k])
            k += 1

        xmldoc1.writexml(
            open('D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/rilsa1_tls.add.xml', 'w', encoding="utf-8"))

    def checkStart(self, start):
        for i in start:
            if i is False:
                return False
        return True

    def giveOutput(self):

        phase0 = []

        a = []
        lanesIds = []
        for i in self.id:
            lanesIds.append(traci.trafficlights.getControlledLanes(i))
            phase0.append(traci.trafficlights.getPhase(i))

        for j in lanesIds:
            for i in range(len(j)):
                data = traci.lane.getLastStepHaltingNumber(j[i])
                a.append(data)

        for i in range(len(a)):
            self.sumHalting[i] += a[i]
        #self.sumHalting = [x + y for x, y in zip(self.sumHalting, a)]

        if self.checkStart(self.starter) is True:

            #data normalization

            for i in range(len(self.sumHalting)):
                self.sumHalting[i] /= 1

            # data normalization
            for i in range(len(self.semaphorePhases)):
                self.semaphorePhases[i] = int(self.semaphorePhases[i])/1

            #laneData = np.asarray(self.sumHalting, dtype=numpy.float64)

            #stateData = np.asarray(self.semaphorePhases, dtype=numpy.float64)
            target_vector = self.predict(self.sumHalting, self.semaphorePhases)

            self.modifyXML(newState=target_vector.tolist())
            for i in self.id:
                traci.trafficlights.setPhase(i, 0)

            #self.semaphorePhases = target_vector.flatten()

            xmldoc = minidom.parse(self.selectedEpisode, )
            itemlist = xmldoc.getElementsByTagName('tlLogic')
            self.semaphorePhases = []

            for item in itemlist:
                power = item.getElementsByTagName('phase')
                phase = []
                for p in power:
                    phase.append(float(p.attributes['duration'].value))
                    nr = 0
                    for s in p.attributes['state'].value:
                        if s is 'g' or s is 'G':
                            nr += 1
                self.semaphorePhases += phase

            for i in range(len(self.starter)):
                self.starter[i] = False

            return target_vector.tolist(), True, sum(self.sumHalting)

        for i in range(len(self.id)):
            if phase0[i] == len(self.originals[i]) - 1:
                self.starter[i] = True
        return self.sumHalting, False

    def fitness(self, scopeAchieved):
        self.fit = scopeAchieved
        return self.fit

    def resetXML(self, selectedEpisode, selectedBackup):
        f = open(selectedBackup, "r")

        lines = f.readlines()

        f2 = open(selectedEpisode, "w")

        f2.writelines(lines)

        f = open("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/backup2.xml", "r")

        lines = f.readlines()

        f2 = open("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/rilsa1_tls.add.xml", "w")

        f2.writelines(lines)

    def boot(self):
        self.resetXML(self.selectedEpisode, self.selectedBackup)
        traci.load(["--start", "-c",
                    self.selectedStartup])

    def run(self):

        self.boot()

        j = 0
        arrived = 0
        deaparted = 0
        fitness = 0

        while j < 1600:
            traci.simulationStep()

            res = self.giveOutput()
            if res[1] == True:
                fitness += res[2]
                self.sumHalting = [0 for _ in self.sumHalting]
                self.modifyXML(res[0])

            j += 1

            arrived += traci.simulation.getArrivedNumber()
            deaparted += traci.simulation.getDepartedNumber()

        print("Results: ", arrived, " ", deaparted, " ", arrived/fitness)
        print(self.fitness(arrived/fitness))
        self.resetXML(selectedEpisode, selectedBackup)

        #traci.close(wait=False)


class Population:
    def __init__(self, sizePopulation):
        self.sizePopulation = sizePopulation
        self.population = [Individual(5, 5) for _ in range(sizePopulation)]
        self.lastBest = 1
        self.currentBest = 1

    def evaluate(self):
        sum = 0
        for x in self.population:
            sum += x.fit
        return sum

    def equationInput(self, parent1, parent2, candidate, Factor):
        list = []
        mutationProb = Factor/2
        for i in range(parent1.neuronsInput):
            l = []
            for j in range(parent1.neuronsHidden):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.inputLayer[i][j] - candidate.inputLayer[i][j]) * Factor + parent1.inputLayer[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.inputLayer[i][j])
            list.append(l)
        return list

    def equationHidden(self, parent1, parent2, candidate, Factor):
        mutationProb = Factor/2
        mutatedLayers = []
        for i in range(parent1.nrHidden-1):
            l = []
            for j in range(parent1.neuronsHidden):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.hiddenLayers[i][j] - candidate.hiddenLayers[i][j]) * Factor + parent1.hiddenLayers[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.hiddenLayers[i][j])
            mutatedLayers.append(numpy.asarray(l))
        return mutatedLayers

    def equationOutput(self, parent1, parent2, candidate, Factor):
        list = []
        mutationProb = Factor/2
        for i in range(parent1.neuronsHidden):
            l = []
            for j in range(parent1.neuronsOutput):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.outputLayer[i][j] - candidate.outputLayer[i][j]) * Factor + parent1.outputLayer[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.outputLayer[i][j])
            list.append(l)
        return list

    def mutate(self, parent1, parent2, candidate):
        donorVector = Individual(parent1.neuronsHidden, parent1.nrHidden)
        factor = 2 * random.uniform(-1, 1) * self.lastBest/(self.currentBest + 0.00001)
        donorVector.inputLayer = numpy.asarray(self.equationInput(parent1, parent2, candidate, factor))
        donorVector.hiddenLayers = numpy.asarray(self.equationHidden(parent1, parent2, candidate, factor))
        donorVector.outputLayer = numpy.asarray(self.equationOutput(parent1, parent2, candidate, factor))

        return donorVector

    def crossover(self, individ1, donorVector):
        crossoverRate = 0.5

        trialVector = Individual(individ1.neuronsHidden, individ1.nrHidden)

        for i in range(len(individ1.inputLayer)):
            for j in range(len(individ1.inputLayer[i])):
                if random.random() > crossoverRate:
                    trialVector.inputLayer[i][j] = individ1.inputLayer[i][j]
                else:
                    trialVector.inputLayer[i][j] = donorVector.inputLayer[i][j]

        for i in range(len(individ1.hiddenLayers)):
            for j in range(len(individ1.hiddenLayers[i])):
                if random.random() > crossoverRate:
                    trialVector.hiddenLayers[i][j] = individ1.hiddenLayers[i][j]
                else:
                    trialVector.hiddenLayers[i][j] = donorVector.hiddenLayers[i][j]

        for i in range(len(individ1.outputLayer)):
            for j in range(len(individ1.outputLayer[i])):
                if random.random() > crossoverRate:
                    trialVector.outputLayer[i][j] = individ1.outputLayer[i][j]
                else:
                    trialVector.outputLayer[i][j] = donorVector.outputLayer[i][j]

        return trialVector

    def evolve(self):
        childred = []
        ind = []
        for i in range(self.sizePopulation):
            self.population[i].run()

        for i in range(self.sizePopulation):

            #candidate = self.population[i]
            parents = random.sample(list(enumerate(self.population)), 3)
            parent1Index, parent1 = parents[0]
            parent2Index, parent2 = parents[1]
            candidateIndex, candidate = parents[2]

            while parent1 == candidate or parent2 == candidate or parent1 == parent2:
                parents = random.sample(self.population, 2)
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

            child = self.mutate(parent1, parent2, candidate)
            childCandidate = self.crossover(candidate, child)
            childred.append(childCandidate)

            ind.append(parent1Index)
            ind.append(parent2Index)
            ind.append(candidateIndex)
            #indexes.append(ind)

        return childred, ind

    def selection(self, children, candidatesIndexes, test=False):
        print(candidatesIndexes)
        j = 0
        for i in range(len(children)):
            #self.population[candidatesIndexes[i][0]].run()
            #self.population[candidatesIndexes[i][1]].run()
            #self.population[candidatesIndexes[i][2]].run()

            children[i].run()

            if self.population[candidatesIndexes[j]].fit <= children[i].fit:
                self.population[candidatesIndexes[j]] = children[i]
            else:
                if self.population[candidatesIndexes[j+1]].fit <= children[i].fit:
                    self.population[candidatesIndexes[j+1]] = children[i]
                else:
                    if self.population[candidatesIndexes[j+2]].fit <= children[i].fit:
                        self.population[candidatesIndexes[j+2]] = children[i]
            j += 3

    def best(self, n):
        aux = sorted(self.population, key=lambda Individual: Individual.fit)
        return aux[:n]


class Cells:
    def __init__(self, newPhases, episode):
        self.semaphorePhases = []
        self.semaphoreStates = []
        self.ids = []
        self.shouldUpdate = []

        self.episode = episode

        for phase in newPhases:
            self.semaphorePhases.append(phase[0])
            self.semaphoreStates.append(phase[1])
            self.ids.append(phase[2])
            self.shouldUpdate.append(phase[3])

    def modifyXML(self, newState, id):

        xmldoc1 = minidom.parse(self.episode)
        k = 0
        for s in xmldoc1.getElementsByTagName('tlLogic'):

            if s.attributes['id'].value == id:

                q = s.getElementsByTagName('phase')

                #k = 0
                for phase in q:
                    phase.attributes['duration'].value = str(newState[0][k])
                    k += 1


        xmldoc1.writexml(
            open(self.episode, 'w', encoding="utf-8"))

        """
        k = 0
        xmldoc1 = minidom.parse("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/rilsa1_tls.add.xml")

        for s in xmldoc1.getElementsByTagName('phase'):
            s.attributes['duration'].value = str(newState[0][k])
            k += 1

        xmldoc1.writexml(
            open('D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/rilsa1_tls.add.xml', 'w', encoding="utf-8"))
        """

    def isCompatible(self, currentState, previousState, nextState):
        compatible = 0

        divider = min(len(currentState), len(previousState), len(nextState))

        currentState = ''.join(currentState)
        previousState = ''.join(previousState)
        nextState = ''.join(nextState)

        size = min(len(currentState), len(previousState), len(nextState))
        for i in range(size):
            if currentState[i] == 'G':
                if currentState[i] == previousState[i] == nextState[i]:
                    compatible += 1

        if compatible >= size // (divider*2):
            return True
        return False

    def modify(self, currentState, previousState, nextState, position):
        currentState = ''.join(currentState)
        previousState = ''.join(previousState)
        nextState = ''.join(nextState)

        size = min(len(currentState), len(previousState), len(nextState))
        for i in range(size):
            if currentState[i] == 'G':
                if currentState[i] == previousState[i] == nextState[i]:
                    l = []
                    if len(self.semaphorePhases[position][0]) < len(self.semaphorePhases[position-1][0]):
                        for i in range(len(self.semaphorePhases[position][0])):
                            if float(self.semaphorePhases[position][0][i]) + float(self.semaphorePhases[position-1][0][i]) < 50:
                                l.append(str(float(self.semaphorePhases[position][0][i]) + float(self.semaphorePhases[position-1][0][i])))
                            else:
                                l.append(str(float(self.semaphorePhases[position][0][i]) + float(self.semaphorePhases[position-1][0][i]) // 2))
                    else:
                        for i in range(len(self.semaphorePhases[position-1][0])):
                            if float(self.semaphorePhases[position][0][i]) + float(self.semaphorePhases[position-1][0][i]) < 50:
                                l.append(str(float(self.semaphorePhases[position][0][i]) + float(self.semaphorePhases[position-1][0][i])))
                            else:
                                l.append(str(float(self.semaphorePhases[position][0][i]) + float(self.semaphorePhases[position-1][0][i]) // 2))
                        cont = i
                        for j in range(cont, len(self.semaphorePhases[position][0])):
                            l.append(self.semaphorePhases[position][0][j])
                    self.semaphorePhases[position][0] = l

    def AutomateIntersections(self):
        for i in range(1, len(self.semaphoreStates) - 1):
            if self.isCompatible(self.semaphoreStates[i][0], self.semaphoreStates[i-1][0],
                                 self.semaphoreStates[i+1][0]):
                self.modify(self.semaphoreStates[i][0], self.semaphoreStates[i-1][0], self.semaphoreStates[i+1][0], i)

        for i in range(len(self.semaphoreStates)):
            if self.shouldUpdate[i] is True:
                self.modifyXML(self.semaphorePhases[i], self.ids[i])



class IntersectionNetwork:

    def __init__(self, params1, params2):
        self.shape1 = None
        self.shape2 = None
        self.model = self.SetModel(params1, params2)

    def SetModel(self, intersectionLanes, semaphorePhases):

        self.shape1 = (1, len(intersectionLanes))
        self.shape2 = (1, len(semaphorePhases[0]))

        input1 = Input(shape=(1, len(intersectionLanes)), name="InputHalting")

        input2 = Input(shape=(1, len(semaphorePhases[0])), name="InputPhases")

        merged_vector = keras.layers.concatenate([input1, input2])

        lstm = LSTM(len(semaphorePhases[0]), activation='relu')(merged_vector)

        dense1 = Dense(len(semaphorePhases[0]), kernel_initializer='lecun_uniform', activation='relu')(lstm)
        dense2 = Dense(len(semaphorePhases[0]), kernel_initializer='lecun_uniform', activation='relu')(dense1)
        dense3 = Dense(len(semaphorePhases[0]), kernel_initializer='lecun_uniform', activation='relu')(dense2)

        output = Dense(len(semaphorePhases[0]), kernel_initializer='lecun_uniform', activation='linear')(dense3)

        model = Model(inputs=[input1, input2],
                      outputs=output)

        rms = RMSprop()
        model.compile(loss='mean_squared_error', optimizer=rms)
        return model


class Agent:
    #sumoBinaryGui = "C:/Program Files (x86)/DLR/Sumo/bin/sumo-gui"

    def __init__(self, item, episode):

        self.episode = episode

        self.id = None

        self.intersections = []
        self.semaphorePhases = []  # actions
        self.semaphoreState = []
        self.semaphoreGreen = []

        self.rewards = []
        self.lanesIds = []

        power = item.getElementsByTagName('phase')
        self.intersections.append(power)

        self.id = item.attributes['id'].value

        phase = []
        state = []
        greenState = []

        for p in power:
            phase.append(p.attributes['duration'].value)
            nr = 0
            for s in p.attributes['state'].value:
                if s is 'g' or s is 'G':
                    nr += 1
            greenState.append(nr)
            state.append(p.attributes['state'].value)
        self.semaphorePhases.append(phase)

        self.rewards.append(0)
        self.semaphoreState.append(state)
        self.semaphoreGreen.append(greenState)

        self.lanesIds = traci.trafficlights.getControlledLanes(self.id)

        self.networks = IntersectionNetwork(self.lanesIds, self.semaphorePhases)

        self.pastIterationPhases = self.semaphorePhases
        self.pastIterationHalting = [[0 for _ in range(len(self.lanesIds))]]
        self.pastIterationState = self.semaphoreState

        self.originals = self.semaphorePhases
        self.currentHalting = []
        self.sumHalting = [0 for _ in range(len(self.lanesIds))]
        self.start = False

    def changeValues(self):
        pass

    def modifyXML(self, newState):

        xmldoc1 = minidom.parse(self.episode)
        q = 0
        for s in xmldoc1.getElementsByTagName('tlLogic'):

            if s.attributes['id'].value == self.id:

                q = s.getElementsByTagName('phase')

                k = 0
                for phase in q:
                    phase.attributes['duration'].value = str(newState[q][k])
                    k += 1
                q += 1


        xmldoc1.writexml(
            open(self.episode, 'w', encoding="utf-8"))

        """
        k = 0
        xmldoc1 = minidom.parse("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/rilsa1_tls.add.xml")

        for s in xmldoc1.getElementsByTagName('phase'):
            s.attributes['duration'].value = str(newState[0][k])
            k += 1

        xmldoc1.writexml(
            open('D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/rilsa1_tls.add.xml', 'w', encoding="utf-8"))
        """

    def run(self):

        phase = traci.trafficlights.getPhase(self.id)

        #print("Intersection: ", self.id, " is in Phase: ", phase)

        a = []
        lanesIds = traci.trafficlights.getControlledLanes(self.id)
        for i in range(len(lanesIds)):
            data = traci.lane.getLastStepHaltingNumber(lanesIds[i])

            # data = np.asarray(data, dtype=np.float64)

            a.append(data)

        for i in range(len(a)):
            self.sumHalting[i] += a[i]

        if self.start:

            self.currentHalting.append(self.sumHalting)

            laneData = np.asarray(self.pastIterationHalting[0]).reshape(1, 1, len(self.pastIterationHalting[0]))
            stateData = np.asarray(self.pastIterationPhases[0]).reshape(1, 1, len(self.pastIterationPhases[0]))
            target_vector = self.networks.model.predict([laneData,
                                                         stateData])

            laneData = np.asarray(self.currentHalting[0]).reshape(1, 1, len(self.currentHalting[0]))
            stateData = np.asarray(self.semaphorePhases[0]).reshape(1, 1, len(self.semaphorePhases[0]))

            if np.random.random() > 0.5:
                reward = np.sum(self.pastIterationHalting[0]) - np.sum(self.currentHalting[0])
                qvalue = np.multiply(0.9, np.max(self.networks.model.predict([laneData,
                                                                stateData])))

                target = reward + qvalue
            else:
                target = np.random.randint(1, len(self.semaphorePhases[0]))

            target_vector[0][np.argmax(self.networks.model.predict([laneData, stateData]))] = target

            for i in range(len(target_vector[0])):
                if target_vector[0][i] < 0:
                    target_vector[0][i] = abs(target_vector[0][i])

                if target_vector[0][i] > 50:
                    target_vector[0][i] = target_vector[0][i]/100

                if target_vector[0][i] < 5:
                    target_vector[0][i] = np.add(target_vector[0][i], len(self.semaphorePhases[0])-1)

            self.networks.model.fit([laneData, stateData], target_vector, epochs=1, verbose=1)

            #self.modifyXML(newState=target_vector)

            traci.trafficlights.setPhase(self.id, 0)

            self.pastIterationHalting = deepcopy(self.currentHalting)
            self.pastIterationPhases = deepcopy(self.semaphorePhases)
            self.pastIterationState = deepcopy(self.semaphoreState)

            self.currentHalting = []

            self.semaphorePhases = target_vector

            self.sumHalting = [0 for _ in range(len(self.lanesIds))]

            #print("HERE ", target_vector)

            self.start = False

            return target_vector.tolist(), self.semaphoreState, self.id, True

        if phase == len(self.originals[0]) - 1:
            self.start = True

        return self.semaphorePhases, self.semaphoreState, self.id, False


selectedEpisode = "D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/cluj-avram-iancu/osm.net.xml"
selectedBackup = "D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/cluj-avram-iancu/backup.xml"
selectedStartup = "D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/cluj-avram-iancu/osm.sumocfg"


xmldoc = minidom.parse(selectedEpisode, )
itemlist = xmldoc.getElementsByTagName('tlLogic')


def modifyXML(selectedEpisode, selectedBackup):
    f = open(selectedBackup, "r")

    lines = f.readlines()

    f2 = open(selectedEpisode, "w")

    f2.writelines(lines)

    f = open("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/backup2.xml", "r")

    lines = f.readlines()

    f2 = open("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/RiLSA_Example/rilsa1_tls.add.xml", "w")

    f2.writelines(lines)

def run(train=False):

    modifyXML(selectedEpisode, selectedBackup)

    arrived = 0
    deaparted = 0
    sumoBinaryGui = "C:/Program Files (x86)/DLR/Sumo/bin/sumo-gui"

    sumo_cmd = [sumoBinaryGui, "--start", "-c",
                selectedStartup]
    program = traci.start(sumo_cmd)

    targetStates = []

    agents = [Agent(item, selectedEpisode) for item in itemlist]

    if train == False:
        for a in agents:
            try:
                a.networks.model.load_weights("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/Weights/weights.h5py")
            except Exception:
                print("This model cannot load these weights")
    res = []
    for episode in range(5):

        j = 0

        while j < 1600:
            traci.simulationStep()

            for a in agents:
                targetStates.append(a.run())

            cells = Cells(targetStates, selectedEpisode)
            cells.AutomateIntersections()

            targetStates = []

            j += 1

            arrived += traci.simulation.getArrivedNumber()
            deaparted += traci.simulation.getDepartedNumber()

        res.append((arrived, deaparted))
        print("Results: ", "Arrived: ", arrived, " Departed: ", deaparted, " Time: ", j)
        arrived = 0
        deaparted = 0
        modifyXML(selectedEpisode, selectedBackup)
        traci.load(["--start", "-c",
                    selectedStartup])
    print(res)

    if train == True:
        for a in agents:
            a.networks.model.save_weights("D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/Weights/weights.h5py")
    traci.close(wait=False)


run(train=False)


"""
sumoBinaryGui = "C:/Program Files (x86)/DLR/Sumo/bin/sumo-gui"
selectedStartup = "D:/facultate/IA/MAdRaPR-Intelligent-Semaphore/cluj-centru-500/osm.sumocfg"

sumo_cmd = [sumoBinaryGui, "--start", "-c",
            selectedStartup]
program = traci.start(sumo_cmd)

def iteration():
    p = Population(30)
    for i in range(100):
        donorVector, indexes = p.evolve()
        p.selection(donorVector, indexes)
        offspringError = p.evaluate()
        p.lastBest = p.currentBest
        print(p.best(1)[0].fit)
        p.currentBest = p.best(1)[0].fit
        print("LOG Global Error")
        print(offspringError)

iteration()
"""