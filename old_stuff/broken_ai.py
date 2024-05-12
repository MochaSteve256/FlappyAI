import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import random
import json

import game


def merge_and_mutate(model1, model2, mutation_rate=0.1):
    """
    Merge parameters of two PyTorch models randomly and apply mutation.

    Args:
        model1 (torch.nn.Module): First PyTorch neural network model.
        model2 (torch.nn.Module): Second PyTorch neural network model.
        mutation_rate (float): Probability of mutation for each parameter.

    Returns:
        torch.nn.Module: Merged and mutated PyTorch neural network model.
    """
    # Create a new model with the same architecture as model1
    merged_model = type(model1)(False)

    # Iterate over the parameters of model1 and model2
    for param1, param2 in zip(merged_model.parameters(), model1.parameters()):
        # take random sample from either model1 or model2
        if random.random() < 0.5:
            param1.data = param1.data + (param2.data - param1.data) * mutation_rate
        else:
            param1.data = param1.data - (param1.data - param2.data) * mutation_rate

    return merged_model



class AI(nn.Module):
    def __init__(self, load_gene):
        super(AI, self).__init__()
        self.i2h = nn.Linear(5, 8)  # Input to first hidden layer
        self.h2h = nn.Linear(8, 8)  # First hidden layer to second hidden layer
        self.h2o = nn.Linear(8, 1)  # Second hidden layer to output
        if load_gene:
            self.load_gene()

    def forward(self, inputs):
        x = F.relu(self.i2h(inputs))  # Apply ReLU activation to input layer to first hidden layer
        x = F.relu(self.h2h(x))  # Apply ReLU activation to first hidden layer to second hidden layer
        x = torch.sigmoid(self.h2o(x))  # Apply Sigmoid activation to second hidden layer to output
        return x
    
    def load_gene(self):
        model1 = AI(False)
        model2 = AI(False)
        model1.load_state_dict(torch.load("ai1.pt"))
        model2.load_state_dict(torch.load("ai2.pt"))
        self.load_state_dict(merge_and_mutate(model1, model2).state_dict())

    def save_gene(self, path):
        torch.save(self.state_dict(), path)

class Instance:
    def __init__(self, load_genes):
        if load_genes:
            self.brain = AI(True)
        else:
            self.brain = AI(False)
        self.inputs = torch.zeros(5)  # 5 input neurons
        self.output = 0.0
        self.bird = game.Bird(False)
        self.holdingJump = False

    def update(self, input, pipeRects):
        self.inputs = torch.tensor([input[0] - self.bird.yPos, input[1] - self.bird.yPos, input[2] - self.bird.xPos, self.bird.yVel, self.bird.yPos], dtype=torch.float32)
        self.output = self.brain(self.inputs)
        if not self.holdingJump and self.output > 0.5:
            birdRect = self.bird.update(True)
        else:
            birdRect = self.bird.update(False)
        self.holdingJump = self.output > 0.5
        colliding = False
        for pipe in pipeRects[0]:
            if birdRect.colliderect(pipe):
                colliding = True
                break
        if birdRect.colliderect(pygame.Rect(0, 640 - 80, 480, 640)):
            colliding = True
        if colliding:
            return False
        return True
    
    def save_gene(self, path):
        self.brain.save_gene(path)

    def render(self, screen):
        self.bird.render(screen)


class instanceManager:
    def __init__(self, instance_count, load_genes_from_file=False):
        self.instances = []
        self.inputs = [0, 0, 0]
        self.initialGenes = []
        self.pipeRects = []
        self.topPipeYpos = 0
        self.bottomPipeYpos = 0
        self.nextPipesXdist = 0
        if load_genes_from_file:
            self.instances.append(Instance(True))
        else:
            for _ in range(instance_count):
                self.instances.append(Instance(False))

    def getInputs(self):
        self.topPipeYpos = self.pipeRects[0][len(self.pipeRects) - 2].h
        self.bottomPipeYpos = self.pipeRects[0][len(self.pipeRects) - 1].y
        self.nextPipesXdist = self.pipeRects[0][len(self.pipeRects) - 1].x - 120
    
    
    def update(self, pipeRects, sessionHighscore, score):
        self.pipeRects = pipeRects
        self.getInputs()
        self.inputs = [self.topPipeYpos, self.bottomPipeYpos, self.nextPipesXdist]
        print("alive: " + str(len(self.instances)))
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
                        try:
                            self.instances[0].save_gene("ai1.pt")
                            self.instances[1].save_gene("ai2.pt")
                            with open("options.json", "w") as f:
                                g = {"aiHighscore": score}
                                json.dump(g, f)
                            
                        except Exception as e:
                            print("Too few instances, using first instance if exists, otherwise randomized genes, exception:", e)
                    else:
                        print("SessionHighscore not exceeded, not saving genes.")
                    return False
        return True

    def get_ai_highscore(self):
        # Calculate and return highscore of AI instances
        highscore = 0
        for instance in self.instances:
            # Assuming there's a method to get the highscore of each instance
            instance_highscore = instance.get_highscore()
            if instance_highscore > highscore:
                highscore = instance_highscore
        return highscore
    
    def render(self, screen):
        for instance in self.instances:
            instance.render(screen)
