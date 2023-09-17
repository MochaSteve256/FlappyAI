# game elements
import pygame
import random

class Bird:
    def __init__(self):
        self.xPos = 120
        self.yPos = 300
        self.yVel = 0
        self.gravity = .06
        self.angle = 0
        path = "assets/sprites/yellowbird-midflap.png"
        self.image = pygame.image.load(path).convert()
        self.image = pygame.transform.scale(self.image, (68, 48))
        self.imageRect = self.image.get_rect()
        self.centerX = self.imageRect.centerx
        self.centerY = self.imageRect.centery
        self.finalImage = self.image
        self.rotatedRect = self.imageRect
        self.afterFlapIter = 0
        self.frame = 1
        self.didAnimate = False
        self.animDelay = 0
    
    def update(self, flap):
        if flap:
            self.yVel = -3.5
            self.angle = 30
            self.afterFlapIter = 0
        else:
            self.yVel += self.gravity
            if self.yVel > 5:
                self.yVel = 5
            if self.afterFlapIter > 50:
                self.angle -= 1
            if self.angle < -90:
                self.angle = -90
            self.afterFlapIter += 1
        
        if self.frame == 0 and not self.didAnimate:
            path = "assets/sprites/yellowbird-midflap.png"
            del self.image
            self.image = pygame.image.load(path).convert()
            self.image = pygame.transform.scale(self.image, (68, 48))
            self.frame = 1
            self.didAnimate = True
        if self.frame == 1 and not self.didAnimate:
            path = "assets/sprites/yellowbird-downflap.png"
            del self.image
            self.image = pygame.image.load(path).convert()
            self.image = pygame.transform.scale(self.image, (68, 48))
            self.frame = 2
            self.didAnimate = True
        if self.frame == 2 and not self.didAnimate:
            path = "assets/sprites/yellowbird-midflap.png"
            del self.image
            self.image = pygame.image.load(path).convert()
            self.image = pygame.transform.scale(self.image, (68, 48))
            self.frame = 3
            self.didAnimate = True
        if self.frame == 3 and not self.didAnimate:
            path = "assets/sprites/yellowbird-upflap.png"
            del self.image
            self.image = pygame.image.load(path).convert()
            self.image = pygame.transform.scale(self.image, (68, 48))
            self.frame = 0
            self.didAnimate = True
        
        if self.animDelay > 15:
            self.didAnimate = False
            self.animDelay = 0
        self.animDelay += 1
        
        if self.yPos < -60:
            self.yPos = -60
        
        self.finalImage = pygame.transform.rotate(self.image, self.angle).convert()
        self.rotatedRect = self.finalImage.get_rect()
        self.rotatedRect.center = (self.centerX + self.xPos, self.centerY + self.yPos)
        self.yPos += self.yVel
        
        return pygame.Rect(self.xPos + 8, self.yPos + 8, 52, 36)
    def render(self, screen):
        screen.blit(self.finalImage, self.rotatedRect)

class Pipe:
    def __init__(self, yPos, isTopPipe, ySpace = 200):
        self.xPos = 480
        if not isTopPipe:
            self.yPos = yPos
            self.image = pygame.image.load("assets/sprites/pipe-green.png").convert()
        else:
            self.yPos = yPos - 320 * 2 - ySpace
            self.image = pygame.image.load("assets/sprites/pipe-green-top.png").convert()
        self.image = pygame.transform.scale(self.image, (104, 640))
    
    def update(self, xSpeed, ySpeed):
        self.xPos -= xSpeed
        self.yPos += ySpeed

        return pygame.Rect(self.xPos, self.yPos, 104, 640)
    def render(self, screen):
        screen.blit(self.image, (self.xPos, self.yPos))

class PipeManager:
    def __init__(self):
        pipeHeight = random.randint(280, 510)
        self.pipes = []
        self.pipes.append(Pipe(pipeHeight, True, 200))
        self.pipes.append(Pipe(pipeHeight, False))
        self.placedPipe = False
    
    def update(self):
        spawnedPipe = False
        if self.pipes[0].xPos < 120:
            if not self.placedPipe:
                self.placedPipe = True
                pipeHeight = random.randint(280, 510)
                self.pipes.append(Pipe(pipeHeight, True, 200))
                self.pipes.append(Pipe(pipeHeight, False))
                spawnedPipe = True

        pipeReturns = []
        for pipe in self.pipes:
            pipeReturns.append(pipe.update(1, 0))
        
        if self.pipes[0].xPos < -104:
            self.pipes.pop(0)
            self.pipes.pop(0)
            self.placedPipe = False
        
        return [pipeReturns, spawnedPipe]
    
    def render(self, screen):
        for pipe in self.pipes:
            pipe.render(screen)

class Background:
    def __init__(self):
        self.image = pygame.image.load("assets/sprites/background-day.png").convert()
        self.image = pygame.transform.scale(self.image, (480, 853))
        self.backgrounds = [0, 480]
    def update(self, xSpeed):
        for i in range(len(self.backgrounds)):
            self.backgrounds[i] -= xSpeed
        if self.backgrounds[0] <= -480:
            self.backgrounds.pop(0)
            self.backgrounds.append(480)
    def render(self, screen):
        for background in self.backgrounds:
            screen.blit(self.image, (background, -100))

class Ground:
    def __init__(self):
        self.image = pygame.image.load("assets/sprites/base.png").convert()
        self.image = pygame.transform.scale(self.image, (480, 160))
        self.grounds = [0, 480]
    def update(self, xSpeed):
        for i in range(len(self.grounds)):
            self.grounds[i] -= xSpeed
        if self.grounds[0] <= -480:
            self.grounds.pop(0)
            self.grounds.append(480)
    def render(self, screen):
        for ground in self.grounds:
            screen.blit(self.image, (ground, 640 - 80))

class Points:
    def __init__(self):
        self.points = 0
        self.font = pygame.font.Font(None, 84)
        self.text = str(self.points)
    def update(self, pts):
        self.points += pts
        self.text = str(self.points)
    def render(self, screen):
        rendered_text = self.font.render(self.text, True, (0, 0, 0))
        screen.blit(rendered_text, (480 // 2 - rendered_text.get_width() // 2, 60))