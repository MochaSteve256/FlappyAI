import ai
import game

import pygame
import json

#main game class
class Game:
    def __init__(self):
        #initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((960, 640), pygame.DOUBLEBUF)
        #self.screen.set_caption("Flappy Bird AI")#doesnt work WTF
        self.clock = pygame.time.Clock()
        self.running = True
        self.gaming = True
        #load config file
        with open("options.json", "r") as f:
            a = json.load(f)
        self.isHuman = a["nextStartupMode"]
        #initialize game elements
        if self.isHuman:
            self.bird = game.Bird(True)
            self.hiSpeed = False
            self.autoRestart = False
        else:
            self.aiManager = ai.instanceManager(1000, True)
            self.hiSpeed = True
            self.autoRestart = True
        self.bg = game.Background()
        self.ground = game.Ground()
        self.ySpace = a["ySpace"]
        self.pipeManager = game.PipeManager(self.ySpace, True)
        self.points = game.Points()
        self.holdingJump = False
        self.shouldFlap = False
        if not self.isHuman:
            with open("ai.json", "r") as g:
                b = json.load(g)
        self.highscore = a["highscore"]
        self.sessionHighscore = 0
        self.font = pygame.font.Font(None, 36)
        if self.isHuman:
            self.modeButton = game.Button("Mode: Human", 500, 80)
        else:
            self.modeButton = game.Button("Mode: AI", 500, 80)
        self.trainModeButton = game.Button("Training Mode: Off", 500, 120)
        self.trainMode = False
        if self.isHuman:
            self.hiSpeedButton = game.Button("HiSpeed: Off", 680, 80)
            self.autoRestartButton = game.Button("AutoRestart: Off", 735, 120)
        else:
            self.hiSpeedButton = game.Button("HiSpeed: On", 680, 80)
            self.autoRestartButton = game.Button("AutoRestart: On", 735, 120)
        self.restarted = False

    def run(self):
        #main game loop
        birdRect = pygame.Rect(0, 0, 0, 0)
        self.restarted = True
        while self.running:
            #quit if user closes window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.modeButton.isHovered():
                        if self.isHuman:
                            self.modeButton.text = "Mode: AI"
                            self.isHuman = 0
                        else:
                            self.modeButton.text = "Mode: Human"
                            self.isHuman = 1
                        self.restarted = False
                        self.gaming = False
                    if self.trainModeButton.isHovered():
                        if self.trainMode:
                            self.trainMode = False
                            self.trainModeButton.text = "Training Mode: Off"
                        else:
                            self.trainMode = True
                            self.trainModeButton.text = "Training Mode: On"
                    if self.hiSpeedButton.isHovered():
                        if self.hiSpeed:
                            self.hiSpeed = False
                            self.hiSpeedButton.text = "HiSpeed: Off"
                        else:
                            self.hiSpeed = True
                            self.hiSpeedButton.text = "HiSpeed: On"
                    if self.autoRestartButton.isHovered():
                        if self.autoRestart:
                            self.autoRestart = False
                            self.autoRestartButton.text = "AutoRestart: Off"
                        else:
                            self.autoRestart = True
                            self.autoRestartButton.text = "AutoRestart: On"
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False
            
            if self.gaming:
                #flap logic
                if self.isHuman:
                    flap = keys[pygame.K_SPACE]
                else:
                    flap = self.shouldFlap
                
                try:
                    if not self.holdingJump and flap:
                        birdRect = self.bird.update(True)
                    else:
                        birdRect = self.bird.update(False)
                    self.holdingJump = flap
                except:
                    pass
                
                self.ground.update(12)
                self.bg.update(5)
                colliding = False
                pipeRects = self.pipeManager.update()
                if self.isHuman:
                    for pipe in pipeRects[0]:
                        if birdRect.colliderect(pipe):
                            colliding = True
                            break
                    if birdRect.colliderect(pygame.Rect(0, 640 -80, 480, 640)):
                        colliding = True
            
                if self.isHuman:
                    if colliding:
                        self.gaming = False
                else:
                    if self.restarted:
                        if not self.aiManager.update(pipeRects, self.sessionHighscore, self.points.points):
                            self.gaming = False
                self.points.update(pipeRects[1])
            
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE] or self.autoRestart:
                    try:
                        del self.bird
                    except:
                        pass
                    try:
                        del self.aiManager
                    except:
                        pass
                    del self.pipeManager
                    del self.points
                    if self.isHuman:
                        self.bird = game.Bird(self.isHuman)
                    else:
                        self.aiManager = ai.instanceManager(100, True)
                    self.pipeManager = game.PipeManager(self.ySpace, True)
                    self.points = game.Points()
                    self.gaming = True
                    self.restarted = True
            
            if self.highscore < self.points.points:
                self.highscore = self.points.points
            if self.sessionHighscore < self.points.points:
                self.sessionHighscore = self.points.points
            
            #clear screen and render stuff
            pygame.draw.rect(self.screen, (92, 210, 255), (0, 0, 480, 640), 0)
            if not self.trainMode:
                #background
                self.bg.render(self.screen)
                #pipes
                self.pipeManager.render(self.screen)
                
                try:
                    #ai birds
                    self.aiManager.render(self.screen)
                except:
                    pass
                try:
                    #bird
                    self.bird.render(self.screen)
                except:
                    pass
                #ground
                self.ground.render(self.screen)
                #points
                self.points.render(self.screen)
                #ai stuff and info
            pygame.draw.rect(self.screen, (0, 0, 0), (480, 0, 480, 640), 0)
            if not self.trainMode:
                if not self.isHuman:
                    self.highscoreText = self.font.render(f"Highscore: {self.sessionHighscore}, AI Highscore: {self.aiManager.getAIHighscore()}", True, (255, 255, 255))
                else:
                    self.highscoreText = self.font.render(f"Highscore: {self.sessionHighscore}", True, (255, 255, 255))
                self.highscore2Text = self.font.render(f"Total Highscore: {self.highscore}", True, (255, 255, 255))
                self.screen.blit(self.highscoreText, (500, 20))
                self.screen.blit(self.highscore2Text, (500, 45))
                self.modeButton.render(self.screen)
                self.hiSpeedButton.render(self.screen)
                self.autoRestartButton.render(self.screen)
            self.trainModeButton.render(self.screen)
            if not self.trainMode:
                pygame.display.flip()
            if not self.hiSpeed:
                self.clock.tick(30)
        self.save(self.highscore, self.isHuman)
        pygame.quit()
    
    def save(self, highscore, nextStartupMode):
        with open("options.json", "w") as f:
            json.dump({
                "highscore": highscore, 
                "nextStartupMode": nextStartupMode,
                "ySpace": self.ySpace
            },
                    f, indent=4)

if __name__ == "__main__":
    flappyBird = Game()
    flappyBird.run()