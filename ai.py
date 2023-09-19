#AI implementation
import math
import game
import random
import pygame

def mergeGenes(a, b):
    outcomes = [1, 2, 3]# 1 = gene a, 2 = gene b, 3 = mutation
    probabilities = [.45, .45, .1]
    result = []
    for i in range(len(a)):
        j = random.choices(outcomes, probabilities)[0]
        if j == 1:
            result.append(a[i])
        elif j == 2:
            result.append(b[i])
        elif j == 3:
            result.append(random.randint(-1000, 1000))#what?
    return result

def convertGenes(gene):
    i2h = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    h2o = [[0], [0], [0], [0]]
    geneIndex = 0
    for i in range(len(i2h)):
        for j in range(len(i2h[i])):
            i2h[i][j] = gene[geneIndex]
            geneIndex += 1
    for i in range(len(h2o)):
        for j in range(len(h2o[i])):
            h2o[i][j] = gene[geneIndex]
            geneIndex += 1
    return (i2h, h2o)

def relu(x):
    #print(x)
    return max(0, x)

def sigmoid(x):
    x = max(-700, min(700, x))
    return 1 / (1 + math.exp(-x))

class ai:
    def __init__(self, gene):
        self.inputNeurons = [0, 0, 0, 0]
        self.hiddenNeurons = [0, 0, 0, 0]
        self.outputNeurons = [0.0]
        self.i2h_params = []
        self.h2o_params = []
        self.gene = gene
        
    def calcOutput(self, inputs):
        a = convertGenes(self.gene)
        self.i2h_params = a[0]
        self.h2o_params = a[1]
        self.inputNeurons = inputs
        for i in range(len(self.hiddenNeurons)):
            # Apply relu to individual elements, not the entire list
            self.hiddenNeurons[i] = relu(
                self.i2h_params[0][i] * self.inputNeurons[0] +
                self.i2h_params[1][i] * self.inputNeurons[1] +
                self.i2h_params[2][i] * self.inputNeurons[2] +
                self.i2h_params[3][i] * self.inputNeurons[3]
            )
        self.outputNeurons[0] = sigmoid(
            self.h2o_params[0][0] * self.hiddenNeurons[0] +
            self.h2o_params[1][0] * self.hiddenNeurons[1] +
            self.h2o_params[2][0] * self.hiddenNeurons[2] +
            self.h2o_params[3][0] * self.hiddenNeurons[3]
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
        self.inputs = [input[0] - self.bird.yPos, input[1] - self.bird.yPos, input[2] - self.bird.xPos, self.bird.yVel]
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
    def __init__(self, instanceCount):
        self.instanceCount = instanceCount
        self.instances = []
        self.inputs = [0, 0, 0]
        self.initialGenes = []
        self.pipeRects = []
        self.topPipeYpos = 0
        self.bottomPipeYpos = 0
        self.nextPipesXdist = 0
        #initial gene population
        for i in range(self.instanceCount):
            self.initialGenes.append([])
            for j in range(20):
                self.initialGenes[i].append(random.randint(-1000, 1000))
            self.instances.append(instance(self.initialGenes[i]))
    def getInputs(self):
        self.topPipeYpos = self.pipeRects[0][len(self.pipeRects) - 2].h
        self.bottomPipeYpos = self.pipeRects[0][len(self.pipeRects) - 1].y
        self.nextPipesXdist = self.pipeRects[0][len(self.pipeRects) - 1].x - 120
    
    def update(self, pipeRects):
        self.pipeRects = pipeRects
        self.getInputs()
        self.inputs = [self.topPipeYpos, self.bottomPipeYpos, self.nextPipesXdist]
        # Iterate over instances in reverse to safely remove them
        for i in range(len(self.instances) - 1, -1, -1):
            if not self.instances[i].update(self.inputs, pipeRects):
                self.instances.pop(i)  # Remove the bird directly
        if len(self.instances) > 0:
            return True
        else:
            return False
    def render(self, screen):
        for instance in self.instances:
            instance.render(screen)

if __name__ == "__main__":
    a = mergeGenes([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    print(a)