# game elements
import pygame

class Bird:
    def __init__(self):
        self.xPos = 100
        self.yPos = 300
        self.yVel = 0
        self.gravity = 1
        path = "assets/sprites/yellowbird-upflap.png"
        self.image = pygame.image.load(path).convert()
        self.image = pygame.transform.scale(self.image, (68, 48))
    
    def update(self, flap):
        if flap:
            self.yVel = -10
        else:
            self.yVel += self.gravity
            if self.yVel > 20:
                self.yVel = 20
        self.yPos += self.yVel
    def render(self, screen):
        screen.blit(self.image, (self.xPos, self.yPos))