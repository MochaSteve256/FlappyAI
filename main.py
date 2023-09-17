import pygame
import game
import ai

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
        #initialize game elements
        self.bird = game.Bird()
        self.bg = game.Background()
        self.ground = game.Ground()
        self.pipeManager = game.PipeManager()
        self.points = game.Points()
        self.holdingJump = False

    def run(self):
        #main game loop
        while self.running:
            #quit if user closes window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False
            
            if self.gaming:
                #jump keypress logic
                if not self.holdingJump and keys[pygame.K_SPACE]:
                    birdRect = self.bird.update(True)
                else:
                    birdRect = self.bird.update(False)
                self.holdingJump = keys[pygame.K_SPACE]
                
                self.ground.update(1)
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
                    self.bird = game.Bird()
                    self.pipeManager = game.PipeManager()
                    self.points = game.Points()
                    self.gaming = True
            
            #clear screen and render stuff
            pygame.draw.rect(self.screen, (92, 210, 255), (0, 0, 480, 640), 0)
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
            ##ai stuff and info
            pygame.draw.rect(self.screen, (0, 0, 0), (480, 0, 480, 640), 0)
            
            
            pygame.display.flip()
            self.clock.tick(200)
        pygame.quit()

if __name__ == "__main__":
    flappyBird = Game()
    flappyBird.run()