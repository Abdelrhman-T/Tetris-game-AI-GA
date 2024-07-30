import random
import numpy as np
import copy
import tetris_base as game

class Chromosome:
    def __init__(self, weights):
        self.weights = weights
        self.score = 0

    def calc_fitness(self, game_state):
        self.score = game_state[2]
        self.fitness = 1 / (1+self.score)

    def calc_best_move(self, board, piece, show_game=False):
        best_X, best_R, best_Y, best_score = 0, 0, 0, -100000

        num_holes_bef, num_blocking_blocks_bef = game.calc_initial_move_info(board)

        for r in range(len(game.PIECES[piece['shape']])):
            for x in range(-2, game.BOARDWIDTH - 2):
                movement_info = game.calc_move_info(board, piece, x, r, num_holes_bef, num_blocking_blocks_bef)

                if movement_info[0]:
                    movement_score = sum(self.weights[i] * movement_info[i] for i in range(0, len(movement_info)))

                    if movement_score > best_score:
                        best_score, best_X, best_R, best_Y = movement_score, x, r, piece['y']

        piece['y'] = best_Y if show_game else -2
        piece['x'], piece['rotation'] = best_X, best_R
        return best_X, best_R

class GA:
    def __init__(self, num_pop, num_weights=8):
        self.chromosomes = [Chromosome(np.random.uniform(-50, 50, size=(num_weights))) for _ in range(num_pop)]
        self.max_weight = []
        self.max_score = 0

        for i, chrom in enumerate(self.chromosomes):
            game_state = game.run_game_AI(chrom, 1000, 200000, True)
            chrom.calc_fitness(game_state)

    def selection(self, population,num_offspring):
        pop = sorted(population, key = lambda x: x.score ,reverse=True)
        best_pop = pop[:num_offspring+1]
        score = np.array([chrom.score for chrom in best_pop])
        total = score.sum()
        prob = (score / total)
        indices = np.random.choice(np.arange(len(best_pop)), size=len(best_pop), p=prob)
        population = np.array(best_pop)
        return population[indices]

    def crossover(self,selected_pop, pc=0.4):
        N_genes = len(selected_pop[0].weights)
        new_chromo = [copy.deepcopy(c) for c in selected_pop]
        cut_point = np.random.randint(1,N_genes-1)
        N_POP = len(selected_pop)

        R = [r for r in np.random.random(size=N_POP)]
        new_chromosomes = [new_chromo[k] for k in range(N_POP) if R[k] < pc]
        for c1 in range(len(new_chromosomes)) :
            temp = new_chromosomes[c1].weights[cut_point:]
            for c2 in range(c1+1,len(new_chromosomes)):
                new_chromosomes[c1].weights[cut_point:] = new_chromosomes[c2].weights[cut_point:]
                new_chromosomes[c2].weights[cut_point:] = temp
        return new_chromo

    def mutation(self,population,mutation_rate=0.01):
        N_genes = len(population[0].weights)
        new_chromo = [copy.deepcopy(c) for c in population]
        point = np.random.randint(0,N_genes)
        for i in range(0, len(population)):
            if random.random() < mutation_rate:
                new_chromo[i].weights[point] = random.uniform(-50,50)
                
        return new_chromo



