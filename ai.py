import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame

import game

class AI(nn.Module):
    def __init__(self, gene):
        super(AI, self).__init__()
        self.i2h = nn.Linear(5, 8)  # Input to first hidden layer
        self.h2h = nn.Linear(8, 8)  # First hidden layer to second hidden layer
        self.h2o = nn.Linear(8, 1)  # Second hidden layer to output
        self.gene = gene

    def forward(self, inputs):
        x = F.relu(self.i2h(inputs))  # Apply ReLU activation to input layer to first hidden layer
        x = F.relu(self.h2h(x))  # Apply ReLU activation to first hidden layer to second hidden layer
        x = torch.sigmoid(self.h2o(x))  # Apply Sigmoid activation to second hidden layer to output
        return x

class Instance:
    def __init__(self, gene):
        self.gene = gene
        self.inputs = torch.zeros(5)  # 5 input neurons
        self.output = 0.0
        self.bird = game.Bird(False)
        self.brain = AI(self.gene)
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

    def render(self, screen):
        self.bird.render(screen)
