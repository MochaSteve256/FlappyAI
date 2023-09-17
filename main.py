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
        #initialize game elements
        self.bird = game.Bird()

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
            
            if keys[pygame.K_SPACE]:
                self.bird.update(True)
            else:
                self.bird.update(False)
            
            #clear screen and render stuff
            self.screen.fill((0, 0, 0))
            pygame.draw.rect(self.screen, (92, 210, 255), (0, 0, 480, 640), 0)
            self.bird.render(self.screen)
            
            
            pygame.display.flip()
            self.clock.tick(75)
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()