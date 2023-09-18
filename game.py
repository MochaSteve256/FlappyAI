# game elements
import pygame
import random

class Bird:
    def __init__(self, isHuman):
        self.xPos = 120
        self.yPos = 300
        self.yVel = 0
        self.gravity = 0.06
        self.angle = 0
        self.isHuman = isHuman
        self.frame = 0
        self.anim_delay = 0
        self.afterFlapIter = 0
        self.load_bird_images()

    def load_bird_images(self):
        bird_colors = ["yellow" if self.isHuman else "blue"]
        self.images = [
            pygame.transform.scale(pygame.image.load(f"assets/sprites/{bird_colors[0]}bird-{flap}flap.png").convert(), (68, 48))
            for flap in ["mid", "down", "mid", "up"]
        ]
        self.finalImage = self.images[self.frame]
        self.imageRect = self.finalImage.get_rect()
        self.centerX = self.imageRect.centerx
        self.centerY = self.imageRect.centery
        self.rotatedRect = self.imageRect

    def update(self, flap):
        if flap:
            self.yVel = -3.5
            self.angle = 30
            self.anim_delay = 0
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

        self.anim_delay += 1

        if self.anim_delay > 15:
            self.frame = (self.frame + 1) % len(self.images)
            self.anim_delay = 0
            
        self.finalImage = pygame.transform.rotate(self.images[self.frame], self.angle).convert()
        
        if self.yPos < -60:
            self.yPos = -60

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
    def __init__(self, ySpace):
        pipeHeight = random.randint(280, 510)
        self.pipes = []
        self.ySpace = ySpace
        self.pipes.append(Pipe(pipeHeight, True, ySpace))
        self.pipes.append(Pipe(pipeHeight, False))
        self.placedPipe = False
    
    def update(self):
        spawnedPipe = False
        if self.pipes[0].xPos < 120:
            if not self.placedPipe:
                self.placedPipe = True
                pipeHeight = random.randint(280, 510)
                self.pipes.append(Pipe(pipeHeight, True, self.ySpace))
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
        self.image = pygame.transform.scale(self.image, (481, 160))
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

class Button:
    def __init__(self, text, xPos, yPos):
        self.font = pygame.font.Font(None, 36)
        self.xPos = xPos
        self.yPos = yPos
        self.text = text
        self.hitbox = pygame.Rect(0, 0, 0, 0)
        self.rendered_text = self.font.render(self.text, True, (255, 255, 255))
    def isHovered(self):
        a = self.xPos < pygame.mouse.get_pos()[0] < self.xPos + self.hitbox.w + 8
        b = self.yPos < pygame.mouse.get_pos()[1] < self.yPos + self.hitbox.h + 8
        if a and b:
            return True
        else:
            return False
    def render(self, screen):
        self.hitbox = pygame.Rect(self.xPos, self.yPos, self.rendered_text.get_width() + 2, self.rendered_text.get_height() + 2)
        if not self.isHovered():
            self.rendered_text = self.font.render(self.text, True, (255, 255, 255))
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(self.xPos, self.yPos, self.rendered_text.get_width() + 8, self.rendered_text.get_height() + 8), 2)
        else:
            self.rendered_text = self.font.render(self.text, True, (0, 0, 0))
            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(self.xPos, self.yPos, self.rendered_text.get_width() + 8, self.rendered_text.get_height() + 8), 0)
        screen.blit(self.rendered_text, (self.xPos + 4, self.yPos + 4))