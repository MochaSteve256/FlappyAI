#AI implementation
import math
import game
import random

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
    return max(0, x)

def sigmoid(x):
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
            self.hiddenNeurons[i] = relu(self.i2h_params[0] * self.inputNeurons[0] + self.i2h_params[1] * self.inputNeurons[1] + self.i2h_params[2] * self.inputNeurons[2] + self.i2h_params[3] * self.inputNeurons[3])
        self.outputNeurons[0] = sigmoid(self.h2o_params[0] * self.hiddenNeurons[0] + self.h2o_params[1] * self.hiddenNeurons[1] + self.h2o_params[2] * self.hiddenNeurons[2] + self.h2o_params[3] * self.hiddenNeurons[3])
        
        return self.outputNeurons[0]
class instance:
    def __init__(self):
        self.bird = game.Bird(False)
        

if __name__ == "__main__":
    a = mergeGenes([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    print(a)