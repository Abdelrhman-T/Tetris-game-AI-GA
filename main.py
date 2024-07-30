import tkinter as tk
from tkinter import messagebox
import GeneticAlgorithm as ga
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
            selected_pop = pop.selection(pop.chromosomes, NUM_OFFSPRING)
            children_chromo = pop.crossover(selected_pop)
            final_chrom = pop.mutation(children_chromo)
            pop.chromosomes.extend(final_chrom)

            for i in range(NUM_OFFSPRING):
                game_state = game.run_game_AI(final_chrom[i], GAME_SPEED, MAX_SCORE, no_show_game)
                final_chrom[i].calc_fitness(game_state)
                if (final_chrom[i].score > pop.max_score):
                    pop.max_score = final_chrom[i].score
                    pop.max_weight = final_chrom[i].weights
                    write_in_file("Best Weight", [e, g, i, pop.max_weight, pop.max_score])

                write_in_file("Train_Score", [e, g, i, final_chrom[i].weights, final_chrom[i].score])

    try:
        plot_best_two_chromo("Train_Score")
    except FileNotFoundError:
        print("File not found. Please ensure that the file exists in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")

    test_iter(pop.max_weight)
    return 0

def start_game(root,player_choice):
    if player_choice == "1":
        root.destroy()
        game.MANUAL_GAME = True
        game.main()
    elif player_choice == "2":
        root.destroy()
        main(False)
    elif player_choice == "3":
        root.destroy()
        max_weight = [-13.76623498, -11.84609247, 45.45997697, -44.74880002, 8.8122277, 41.19293833, -34.70749797, 33.75808344]
        test_iter(max_weight)
    else:
        messagebox.showerror("Invalid Selection", "Please select a valid option.")

def main_menu():
    root = tk.Tk()
    root.title("Tetris AI Trainer")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = int(screen_width * 0.23)
    window_height = int(screen_height * 0.3)
    root.geometry(f"{window_width}x{window_height}")

    position_x = (screen_width - window_width) // 2
    position_y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

    root.configure(bg='black')

    # Configure Label and Button styles
    label = tk.Label(root, text="Who do you want to play?", bg='black', fg='red', font=('Arial', 20))
    label.pack(pady=10)

    button_style = {'bg': 'black', 'fg': 'white', 'font': ('Arial', 14)}
    tk.Button(root, text="1- User", command=lambda: start_game(root,"1"), **button_style).pack(fill=tk.X, pady=5)
    tk.Button(root, text="2- Train AI then play", command=lambda: start_game(root,"2"), **button_style).pack(fill=tk.X, pady=5)
    tk.Button(root, text="3- AI play", command=lambda: start_game(root,"3"), **button_style).pack(fill=tk.X, pady=5)

    root.mainloop()

if __name__ == "__main__":
    main_menu()
