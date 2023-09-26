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
        self.hardcore = False
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
        self.pipeManager = game.PipeManager(self.ySpace, self.hardcore)
        self.points = game.Points()
        self.holdingJump = False
        self.shouldFlap = False
        self.fitness = 0
        self.aiFitness = 0
        if not self.isHuman:
            with open("ai.json", "r") as g:
                b = json.load(g)
                self.aiFitness = b["aiFitness"]
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
        self.resetAI = game.Button("Reset AI", 845, 80)

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
                            self.hiSpeed = True
                            self.hiSpeedButton.text = "HiSpeed: On"
                            self.autoRestart = 1
                            self.autoRestartButton.text = "AutoRestart: On"
                        else:
                            self.modeButton.text = "Mode: Human"
                            self.isHuman = 1
                            self.hiSpeed = False
                            self.hiSpeedButton.text = "HiSpeed: Off"
                            self.autoRestart = 0
                            self.autoRestartButton.text = "AutoRestart: Off"
                        self.restarted = False
                        self.gaming = False
                    if self.trainModeButton.isHovered():
                        if self.trainMode:
                            self.trainMode = False
                            self.trainModeButton.text = "Training Mode: Off"
                        else:
                            self.trainMode = True
                            self.trainModeButton.text = "Training Mode: On"
                            self.trainModeButton.render(self.screen)
                            pygame.display.flip()
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
                    if self.resetAI.isHovered():
                        with open("ai.json", "r") as f:
                            a = json.load(f)
                        a["aiFitness"] = 0
                        with open("ai.json", "w") as f:
                            json.dump(a, f, indent=4)
                        self.running = False
            
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
                
                self.ground.update(1.2)
                self.bg.update(.5)
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
                        if not self.aiManager.update(pipeRects, self.aiFitness, self.fitness):
                            self.gaming = False
                self.points.update(pipeRects[1])
                self.fitness += 1
            
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
                        self.aiManager = ai.instanceManager(200, True)
                    self.pipeManager = game.PipeManager(self.ySpace, self.hardcore)
                    self.points = game.Points()
                    self.gaming = True
                    self.restarted = True
                    self.fitness = 0
            
            if self.highscore < self.points.points:
                self.highscore = self.points.points
            if self.sessionHighscore < self.points.points:
                self.sessionHighscore = self.points.points
            if self.fitness > self.aiFitness:
                self.aiFitness = self.fitness
            
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
                    try:
                        self.highscoreText = self.font.render(f"Highscore: {self.sessionHighscore}, AI fitness: {self.aiManager.getAIHighscore()}", True, (255, 255, 255))
                        self.alive = self.font.render(f"Alive: {len(self.aiManager.instances)}, 1 Score = 360 Fitness", True, (255, 255, 255))
                        self.screen.blit(self.alive, (500, 157))
                    except:
                        pass
                else:
                    self.highscoreText = self.font.render(f"Highscore: {self.sessionHighscore}", True, (255, 255, 255))
                self.highscore2Text = self.font.render(f"Total Highscore: {self.highscore}", True, (255, 255, 255))
                self.screen.blit(self.highscore2Text, (500, 45))
                try:
                    self.screen.blit(self.highscoreText, (500, 20))
                    
                except:
                    pass
                self.modeButton.render(self.screen)
                self.hiSpeedButton.render(self.screen)
                self.autoRestartButton.render(self.screen)
                self.resetAI.render(self.screen)
            self.trainModeButton.render(self.screen)
            if not self.trainMode:
                pygame.display.flip()
            if not self.hiSpeed:
                self.clock.tick(200)
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