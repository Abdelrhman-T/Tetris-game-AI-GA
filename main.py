import  GeneticAlgorithm as ga
import tetris_base as game
from helper_function import write_in_file
from helper_function import test_iter
from helper_function import plot_best_two_chromo
import copy

GAME_SPEED = 100000
NUM_GENERATION = 20
NUM_POP = 30
TRAIN_ITER = 300
NUM_OFFSPRING = 10
MAX_SCORE = 100000
NUM_WIN = 0


def main(no_show_game):

    write_in_file("Train_Score", ["Iteration","Generation","chromosome","weights","score"])
    write_in_file("Best Weight",["Iteration","Generation","chromosome","weights","score"])

    # Initialize population
    init_pop = ga.GA(NUM_POP)
    pop = copy.deepcopy(init_pop)

    for e in range(TRAIN_ITER):

        for g in range(NUM_GENERATION):
            
            selected_pop = pop.selection(pop.chromosomes)

            children_chromo = pop.crossover(selected_pop)
            
            final_chrom = pop.mutation(children_chromo)
            
            for i in range(NUM_OFFSPRING):

                # Run the game for each chromosome
                game_state = game.run_game_AI(final_chrom[i], GAME_SPEED,
                                            MAX_SCORE, no_show_game)
                # Calculate the fitness
                final_chrom[i].calc_fitness(game_state)
                if (final_chrom[i].score > pop.max_score):
                    pop.max_score = final_chrom[i].score
                    pop.max_weight = final_chrom[i].weights
                    
                    write_in_file("Best Weight",[e,g,i,pop.max_weight,pop.max_score])

                write_in_file("Train_Score",[e,g,i,final_chrom[i].weights,final_chrom[i].score])

    try:
        plot_best_two_chromo("Train_Score")
    except FileNotFoundError:
        print("File not found. Please ensure that the file exists in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    test_iter(pop.max_weight)

    return 0

if __name__ == "__main__":
    palyer = input("Who do you want to play? \n 1- user \n 2- AI(Train Then Test)\n 3- AI(Test) \n===> ")

    if (palyer=="1"):
        game.MANUAL_GAME = True
        game.main()
    
    elif (palyer=="2"):
        main(True)

    elif (palyer=="3"):
        # 1st
        # [-13.76623498 -11.84609247  45.45997697 -44.74880002   8.8122277, 41.19293833 -34.70749797  33.75808344]
        max_weight = [-13.76623498, -11.84609247, 45.45997697, -44.74880002, 8.8122277, 41.19293833, -34.70749797, 33.75808344]
        #2nd
        # max_weight = [-13.76623498, -11.84609247,  45.45997697, -44.74880002, 8.8122277, 34.73571674, -28.62565297, 39.33287464]
        test_iter(max_weight)