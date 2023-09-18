import pygame
import game
import ai
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
        self.bird = game.Bird(self.isHuman)
        self.bg = game.Background()
        self.ground = game.Ground()
        self.ySpace = a["ySpace"]
        self.pipeManager = game.PipeManager(self.ySpace)
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
        self.hiSpeed = False
        self.hiSpeedButton = game.Button("HiSpeed: Off", 680, 80)

    def run(self):
        #main game loop
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
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False
            
            if self.gaming:
                #flap logic
                if self.isHuman:
                    flap = keys[pygame.K_SPACE]
                else:
                    flap = self.shouldFlap
                
                if not self.holdingJump and flap:
                    birdRect = self.bird.update(True)
                else:
                    birdRect = self.bird.update(False)
                self.holdingJump = flap
                
                self.ground.update(1.2)
                self.bg.update(.5)
                pipeRects = self.pipeManager.update()
                colliding = False
                for pipe in pipeRects[0]:
                    if birdRect.colliderect(pipe):
                        colliding = True
                        break
                if birdRect.colliderect(pygame.Rect(0, 640 -80, 480, 640)):
                    colliding = True
            
                if colliding:
                    self.gaming = False
            
                self.points.update(pipeRects[1])
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    del self.bird
                    del self.pipeManager
                    del self.points
                    self.bird = game.Bird(self.isHuman)
                    self.pipeManager = game.PipeManager(self.ySpace)
                    self.points = game.Points()
                    self.gaming = True
            
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
                #bird
                self.bird.render(self.screen)
                #ground
                self.ground.render(self.screen)
                #points
                self.points.render(self.screen)
                #ai stuff and info
            pygame.draw.rect(self.screen, (0, 0, 0), (480, 0, 480, 640), 0)
            if not self.trainMode:
                self.highscoreText = self.font.render(f"Session Highscore: {self.sessionHighscore}", True, (255, 255, 255))
                self.highscore2Text = self.font.render(f"Total Highscore: {self.highscore}", True, (255, 255, 255))
                self.screen.blit(self.highscoreText, (500, 20))
                self.screen.blit(self.highscore2Text, (500, 45))
                self.modeButton.render(self.screen)
                self.hiSpeedButton.render(self.screen)
            self.trainModeButton.render(self.screen)
            
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
            }, f, indent=4)
        '''
        if not self.isHuman:
            with open("ai.json", "w") as g:
                json.dump({"best1": 1, "best2": 1}, g)
        '''

if __name__ == "__main__":
    flappyBird = Game()
    flappyBird.run()