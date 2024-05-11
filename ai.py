#AI implementation
import game

import pygame
import math
import random
import json

def mergeGenes(a, b):
    outcomes = [1, 2, 3]# 1 = gene a, 2 = gene b, 3 = mutation
    probabilities = [2 / 5, 2 / 5, 1 / 5]
    result = []
    for i in range(len(a)):
        j = random.choices(outcomes, probabilities)[0]
        if j == 1:
            result.append(a[i])
        elif j == 2:
            result.append(b[i])
        elif j == 3:
            if random.randint(0, 2) == 0:
                result.append(random.randint(-1000, 1000))
            else:
                if random.randint(0, 1):
                    result.append(a[i] + random.randint(-50, 50))
                else:
                    result.append(b[i] + random.randint(-50, 50))
    return result

def convertGenes(gene):
    i2h = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    h2h = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    h2o = [[0], [0], [0], [0], [0], [0], [0], [0]]
    geneIndex = 0
    for i in range(len(i2h)):
        for j in range(len(i2h[i])):
            i2h[i][j] = gene[geneIndex]
            geneIndex += 1
    for i in range(len(h2h)):
        for j in range(len(h2h[i])):
            h2h[i][j] = gene[geneIndex]
            geneIndex += 1
    for i in range(len(h2o)):
        for j in range(len(h2o[i])):
            h2o[i][j] = gene[geneIndex]
            geneIndex += 1
    return (i2h, h2h, h2o)

def relu(x):
    #print(x)
    return math.tanh(x)

def sigmoid(x):
    x = max(-700, min(700, x))
    return 1 / (1 + math.exp(-x))

# ai currently too dumb, needs more neurons
#plan: 3/4 inputs (maybe rm xDist bc useless), 2 hidden layers 6 neurons each, 1 output for flapping
class ai:
    def __init__(self, gene):
        self.inputNeurons = [0, 0, 0, 0, 0]
        self.hiddenNeurons = [.0, .0, .0, .0, .0, .0, .0, .0]
        self.hiddenNeurons2 = [.0, .0, .0, .0, .0, .0, .0, .0]
        self.outputNeurons = [.0]
        self.i2h_params = []
        self.h2o_params = []
        self.gene = gene
        
    def calcOutput(self, inputs):
        a = convertGenes(self.gene)
        self.i2h_params = a[0]
        self.h2h_params = a[1]
        self.h2o_params = a[2]
        self.inputNeurons = inputs
        for i in range(len(self.hiddenNeurons)):
            # do de ai stuff
            self.hiddenNeurons[i] = relu(
                self.i2h_params[0][i] * self.inputNeurons[0] +#yDist topPipe
                self.i2h_params[1][i] * self.inputNeurons[1] +#yDist bottomPipe
                self.i2h_params[2][i] * self.inputNeurons[2] +#yVel Bird
                self.i2h_params[3][i] * self.inputNeurons[3] +#xDist nextPipe
                self.i2h_params[4][i] * self.inputNeurons[4]  #yPos Bird
            )
        for i in range(len(self.hiddenNeurons2)):
            self.hiddenNeurons2[i] = relu(
                self.h2h_params[0][i] * self.hiddenNeurons[0] +
                self.h2h_params[1][i] * self.hiddenNeurons[1] +
                self.h2h_params[2][i] * self.hiddenNeurons[2] +
                self.h2h_params[3][i] * self.hiddenNeurons[3] +
                self.h2h_params[4][i] * self.hiddenNeurons[4] +
                self.h2h_params[5][i] * self.hiddenNeurons[5] +
                self.h2h_params[6][i] * self.hiddenNeurons[6] +
                self.h2h_params[7][i] * self.hiddenNeurons[7]
            )
        self.outputNeurons[0] = sigmoid(
            self.h2o_params[0][0] * self.hiddenNeurons2[0] +
            self.h2o_params[1][0] * self.hiddenNeurons2[1] +
            self.h2o_params[2][0] * self.hiddenNeurons2[2] +
            self.h2o_params[3][0] * self.hiddenNeurons2[3] +
            self.h2o_params[4][0] * self.hiddenNeurons2[4] +
            self.h2o_params[5][0] * self.hiddenNeurons2[5] +
            self.h2o_params[6][0] * self.hiddenNeurons2[6] +
            self.h2o_params[7][0] * self.hiddenNeurons2[7]
        )

        return self.outputNeurons[0]

class instance:
    def __init__(self, gene):
        self.gene = gene
        # 0: yDist to bottom end of top pipe, 1: yDist to top end of bottom pipe, 2: xDist to left side of a pipe
        self.inputs = [0, 0, 0]
        self.output = 0.0
        self.bird = game.Bird(False)
        self.brain = ai(self.gene)
        self.holdingJump = False
    def update(self, input, pipeRects):
        """
        Params:
            Input: [0] = yPos of bottom of top pipe, [1] = yPos of top of bottom pipe, [2] = xDist to left side of a pipe

        Returns:
            birdRect
        """
        self.inputs = [input[0] - self.bird.yPos, input[1] - self.bird.yPos, input[2] - self.bird.xPos, self.bird.yVel, self.bird.yPos]
        self.output = self.brain.calcOutput(self.inputs)
        if not self.holdingJump and self.output > .5:
            birdRect = self.bird.update(True)
        else:
            birdRect = self.bird.update(False)
        self.holdingJump = self.output > .5
        colliding = False
        for pipe in pipeRects[0]:
            if birdRect.colliderect(pipe):
                colliding = True
                break
        if birdRect.colliderect(pygame.Rect(0, 640 -80, 480, 640)):
            colliding = True
        if colliding:
            return False
        return True
    def render(self, screen):
        self.bird.render(screen)

class instanceManager:
    def __init__(self, instanceCount, loadGenesFromFile):
        self.instanceCount = instanceCount
        self.instances = []
        self.inputs = [0, 0, 0]
        self.initialGenes = []
        self.pipeRects = []
        self.topPipeYpos = 0
        self.bottomPipeYpos = 0
        self.nextPipesXdist = 0
        self.aiHighscore = json.load(open("ai.json", "r"))["aiHighscore"]
        if not loadGenesFromFile:
            #initial gene population
            for i in range(self.instanceCount):
                self.initialGenes.append([])
                for j in range(5*8+8*8+8*1):
                    self.initialGenes[i].append(random.randint(-1000, 1000))
                self.instances.append(instance(self.initialGenes[i]))
        else:
            with open("ai.json", "r") as f:
                self.bestGenes = json.load(f)
            for i in range(self.instanceCount):
                if len(self.bestGenes) >= 2:
                    self.initialGenes.append(mergeGenes(self.bestGenes["bestInstanceA"], self.bestGenes["bestInstanceB"]))
                elif len(self.bestGenes) == 1:
                    self.initialGenes.append(mergeGenes(self.bestGenes["bestInstanceA"], self.bestGenes["bestInstanceA"]))
                self.instances.append(instance(self.initialGenes[i]))
    def getInputs(self):
        self.topPipeYpos = self.pipeRects[0][len(self.pipeRects) - 2].h
        self.bottomPipeYpos = self.pipeRects[0][len(self.pipeRects) - 1].y
        self.nextPipesXdist = self.pipeRects[0][len(self.pipeRects) - 1].x - 120
    
    def update(self, pipeRects, sessionHighscore, score):
        self.pipeRects = pipeRects
        self.getInputs()
        self.inputs = [self.topPipeYpos, self.bottomPipeYpos, self.nextPipesXdist]
        # iterate over instances in reverse to remove w/o m the wrong one bc number changed
        for i in range(len(self.instances) - 1, -1, -1):
            if not self.instances[i].update(self.inputs, pipeRects):
                if len(self.instances) > 2:
                    self.instances.pop(i)  # let bird be garbage-collected by python
                else:
                    self.aiHighscore = json.load(open("ai.json", "r"))["aiHighscore"]
                    if score >= sessionHighscore:
                        #save premium dna
                        print("\033[92mTraining progressed, saving genes\033[0m")
                        bestInstances = {}
                        try:
                            bestInstances["bestInstanceA"] = self.instances[0].gene
                            bestInstances["bestInstanceB"] = self.instances[1].gene
                            bestInstances["aiHighscore"] = score
                        except Exception as e:
                            print("Too few instances, using first instance if even existing, or randomized genes")
                        if len(bestInstances) == 0:
                            bestInstances["bestInstanceA"] = [random.randint(-1000, 1000) for i in range(20)]
                        with open("ai.json", "w") as f:
                            json.dump(bestInstances, f, indent=4)
                    else:
                        print("SessionHighscore not exceeded, not saving genes.")
                    return False
        return True
    def render(self, screen):
        for instance in self.instances:
            instance.render(screen)
        
    def getAIHighscore(self):
        return self.aiHighscore
