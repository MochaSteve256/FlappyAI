import game

import pygame
import neat

class InstanceManager:
    def __init__(self):
        # init vars
        self.inputs = []
        self.genomes = []
        self.nets = []
        self.birds = []
        self.pipeRects = []
        self.topPipeYpos = 0
        self.bottomPipeYpos = 0
        self.nextPipesXdist = 0
        # Initialize NEAT configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'neat_ai.cfg')
        
        p = neat.Population(config)
        
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        winner = p.run(self.eval_genomes, 50)
        
        best_network = neat.nn.FeedForwardNetwork.create(winner, config)
    
    def get_nets_outputs(self, nets, inputs):
        outputs = []
        for net in nets:
            outputs.append(net.activate(inputs))
        return outputs
        
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            # Create and evaluate the neural network
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.nets.append(net)
            self.birds.append(game.Bird(False))
            self.genomes.append(genome)

            # Reset game variables
            self.pipeRects = []  # Reset pipe rects for new evaluation
            score = 0  # Reset score

            # Game loop for this bird
            while any(not bird.dead for bird in self.birds):
                # Perform game update
                self.update(self.pipeRects, score)
                score += 1  # Increment score as time progresses

                # Render the game (optional)
                # self.render(screen)

            # All birds died, assign fitness to genome
            for i in range(len(self.birds)):
                self.genomes[i].fitness = score if not self.birds[i].dead else 0

            # Clear the lists for the next evaluation
            self.pipeRects.clear()
            self.birds.clear()
            self.nets.clear()

    def getInputs(self):
        self.topPipeYpos = self.pipeRects[0][len(self.pipeRects) - 2].h
        self.bottomPipeYpos = self.pipeRects[0][len(self.pipeRects) - 1].y
        self.nextPipesXdist = self.pipeRects[0][len(self.pipeRects) - 1].x - 120
    
    def update(self, pipeRects, score):
        self.pipeRects = pipeRects
        self.getInputs()
        self.inputs = [self.topPipeYpos, self.bottomPipeYpos, self.nextPipesXdist]
        outputs = self.get_nets_outputs(self.nets, self.inputs)

        # Iterate over birds in reverse order
        for i in range(len(self.birds) - 1, -1, -1):
            bird = self.birds[i]
            birdRect = 0
            if outputs[i] > .5:
                birdRect = bird.update(True)
            else:
                birdRect = bird.update(False)

            colliding = False
            for pipe in self.pipeRects[0]:
                if birdRect.colliderect(pipe):
                    colliding = True
                    break
            if birdRect.colliderect(pygame.Rect(0, 640 - 80, 480, 640)):
                colliding = True
            if colliding:
                # Pass fitness to genome
                self.genomes[i].fitness = score
                self.birds[i].dead = True

    def get_ai_highscore(self):
        return 0

    def render(self, screen):
        for bird in self.birds:
            bird.render(screen)