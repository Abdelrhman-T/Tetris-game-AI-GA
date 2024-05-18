import matplotlib.pyplot as plt
import numpy as np

TEST_ITER = 600
TRAIN_ITER = 300
MAX_SCORE = 100000
import  GeneticAlgorithm as ga
import tetris_base as game
SCORE_Arr = []
import csv
import pandas as pd


def write_in_file(file, data):
    with open(f'log/{file}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def plot_test(data):
    df = pd.read_csv(f'log/{data}.csv')
    iteration = list(range(0, 600))
    plt.figure(figsize=(10, 6))
    plt.plot(iteration, df["score"], label='Test Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('The progress of the Test')
    plt.legend(loc='upper center', bbox_to_anchor=(0.67, 0.97))
    plt.grid(True)
    plt.savefig('Graphs\\Test Progress')
    plt.show()



def get_top_chromosomes(df):
    sorted_df = df.sort_values(by='score', ascending=False)
    top_chromosomes = sorted_df[:2]["chromosome"]
    return np.array(top_chromosomes)



def select_scores(df):
    top_chromosomes = get_top_chromosomes(df)
    print("Top chromosomes:", top_chromosomes)
    top_scores1 = df[df['chromosome'] == top_chromosomes[0]]["score"]
    top_scores2 = df[df['chromosome'] == top_chromosomes[1]]["score"]
    return top_scores1, top_chromosomes[0], top_scores2, top_chromosomes[1]


def plot_best_two_chromo(data):
    df = pd.read_csv(f'log/{data}.csv')
    chrom1, c1, chrom2, c2 = select_scores(df)
    states = list(range(0, max(len(chrom1), len(chrom2))))
    plt.figure(figsize=(10, 6))
    plt.plot(states, chrom1, label=f'chromosome {c1}')
    plt.plot(states, chrom2, label=f'chromosome {c2}')
    plt.xlabel('states')
    plt.ylabel('Score')
    plt.title('The progress of best Two chromosomes')
    plt.legend(loc='upper center', bbox_to_anchor=(0.67, 0.97))
    plt.grid(True)
    plt.savefig('Graphs\\best_two_chromosomes_plot')
    plt.show()



def plot_best_two_chromo_seasonality(data):
    df = pd.read_csv(f'log/{data}.csv')
    chrom1, c1, chrom2, c2 = select_scores(df)
    chrom1 = chrom1.rolling(200).mean()
    chrom2 = chrom2.rolling(200).mean()
    states = list(range(0, max(len(chrom1), len(chrom2))))
    plt.figure(figsize=(15, 4))
    plt.plot(states, chrom1, label=f'chromosome {c1}' ,linewidth=1)
    plt.plot(states, chrom2, label=f'chromosome {c2}',  linewidth=1)
    plt.xlabel('states')
    plt.ylabel('Score')
    plt.title('The progress of best Two chromosomes Seasonality')
    plt.legend(loc='upper center', bbox_to_anchor=(0.67, 0.97))
    plt.grid(True)
    plt.savefig('Graphs\\best_two_chromosomes_seasonality_plot.png')
    plt.show()



def test_iter(max_weight):
    num_win = 0
    write_in_file("Test_Score", ["Iteration","score"])

    for i in range(TEST_ITER):

        print(f'\n - - - - Test: {i+1} - - - - \n')

        optimal_weights = max_weight
        chromo = ga.Chromosome(optimal_weights)
        game_state = game.run_game_AI(chromo, speed=600, max_score=100000, no_show=False)
        SCORE_Arr.append(game_state[2])
        
        write_in_file("Test_Score", [i,game_state[2]])

        if (MAX_SCORE <= game_state[2]):
            num_win += 1
            game.display_message("You Win ;)")

    
    print("Test_Score",f'You Win {num_win} of Total {TEST_ITER}')
    print("Test_Score",f'The Accuracy = {num_win/TEST_ITER *100 :.2f}')

    plot_test("Test_Score")



# plot_best_two_chromo_seasonality("Train_Score")
# plot_best_two_chromo("Train_Score")