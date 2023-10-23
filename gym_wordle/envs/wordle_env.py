import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pkg_resources # The pkg_resources module distributed with setuptools provides an API for Python libraries to access their resource files
import random # Generates pseudorandom numbers
from typing import Optional # This module provides runtime support for type hints  https://docs.python.org/3/library/typing.html
import colorama # Used for producing colored terminal text across multiple platforms (e.g. Windows) https://pypi.org/project/colorama/
from colorama import Fore
from colorama import Style
# Cross-platform printing of colored text can then be done using Colorama’s constant shorthand for ANSI escape sequences. Can try uncommenting and running file.
# print(Fore.RED + 'some red text')
# print(Style.DIM + 'and in dim text')
# print(Style.RESET_ALL)
# print('back to normal now')

from gym_wordle.exceptions import InvalidWordException

colorama.init(autoreset=True)

# global game variables
GAME_LENGTH = 6
WORD_LENGTH = 5

# load words and then encode
filename = pkg_resources.resource_filename(
    'gym_wordle',
    'data/5_words.txt'
) # an easier way would be to say filename = '../data/5_words.txt'

# Opposite of strToEncode?
def encodeToStr(encoding):
    string = ""
    for enc in encoding:
        string += chr(ord('a') + enc)
    return string

# Converts an array of 5-letter words to an array of 5-number-tuples
def strToEncode(lines): # lines is ["aback", "abase", ...]
    encoding = []
    for line in lines:
        assert len(line.strip()) == 5  # Must contain 5-letter words for now
        encoding.append(tuple(ord(char) - 97 for char in line.strip()))
    return encoding # this return variable is an array of tuples: [(0, 1, 0, 2, 10), (0, 1, 0, 18, 4), ...] (note the 2 tuples I showed correspond to 'aback' and 'abase')


with open(filename, "r") as f:
    WORDS = strToEncode(f.readlines()) # f.readlines() reads the contents of the file f line by line and returns a list where each element is a line from the file.


class WordleEnv(gym.Env):
    """
    Simple Wordle Environment

    Wordle is a guessing game where the player has 6 guesses to guess the
    5 letter hidden word. After each guess, the player gets feedback on the
    board regarding the word guessed. For each character in the guessed word:
        * if the character is not in the hidden word, the character is
          grayed out (encoded as 0 in the environment)
        * if the character is in the hidden word but not in correct
          location, the character is yellowed out (encoded as 1 in the
          environment)
        * if the character is in the hidden word and in the correct
          location, the character is greened out (encoded as 2 in the
          environment)

    The player continues to guess until they have either guessed the correct
    hidden word, or they have run out of guesses.

    The environment is structured in the following way:
        * Action Space: the action space is a length 5 MulitDiscrete where valid values
          are [0, 25], corresponding to characters [a, z].
        * Observation Space: the observation space is dict consisting of
          two objects:
          - board: The board is 6x5 Box corresponding to the history of
            guesses. At the start of the game, the board is filled entirely
            with -1 values, indicating no guess has been made. As the player
            guesses words, the rows will fill up with values in the range
            [0, 2] indicating whether the characters are missing in the
            hidden word, in the incorrect position, or in the correct position
			based on the most recent guess.
          - alphabet: the alphabet is a length 26 Box corresponding to the guess status
            for each letter in the alaphabet. As the start, all values are -1, as no letter
            has been used in a guess. As the player guesses words, the letters in the
            alphabet will change to values in the range [0, 2] indicating whether the
            characters are missing in the hidden word, in the incorrect position,
            or in the correct position.
    """

    def __init__(self):
        super(WordleEnv, self).__init__() # calls the constructor of the parent class, gym.Env

        # DISCRETE IS ONE OF THE FUNDAMENTAL SPACES
        # https://gymnasium.farama.org/api/spaces/fundamental/#discrete
        # It's a space consisting of finitely many elements.
        # e.g. observation_space = Discrete(2) # {0, 1}
        #      observation_space.sample() # 0

        # MULTIDISCRETE IS ONE OF THE FUNDAMENTAL SPACES
        # https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete
        # A nintendo controller can be conceptualized as 3 discrete action spaces:
        # 1) Arrow Keys: 0 [None], 1 [Up], 2[Right], 3[Down], 4[Left] 
        # 2) Button A: 0 [None], 1 [Pressed] 
        # 3) Button B: 0 [None], 1 [Pressed] 
        # You can represent a given action as spaces.MultiDiscrete([ 5, 2, 2 ]). A sample might be array([3, 1, 0])
        self.action_space = spaces.MultiDiscrete([26] * WORD_LENGTH)

        # BOX IS ONE OF THE FUNDAMENTAL SPACES
        # It's a (possibly unbounded) box in R^n. 
        # Imagine a 2D square on a graph with bottom LHS corner at (1, 1) & top RHS corner at (2, 2)
        # The shape of the space is 2D space with "individual" elements, so we say (I think): shape=(2,)
        # Box(low=-1.0, high=2.0, shape=(2, ), dtype=np.float32)
        # Sampling from the box is kinda like picking real-valued coordinates in the space.

        # DICT IS ONE OF THE COMPOSITE SPACES
        # The keys of a dict are whatever you like, the values are one of the fundamental (or composite!) spaces
        # e.g. observation_space = Dict({"colorA": Discrete(3), "colorB": Discrete(3)})
        #      observation_space.sample() # OrderedDict([('colorA', 2), ('colorB', 0)])
        # Elements of this space are (ordered) dictionaries of elements from the constituent spaces.
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=2, shape=(GAME_LENGTH, WORD_LENGTH), dtype=int),
            'alphabet': spaces.Box(low=-1, high=2, shape=(26,), dtype=int)
        })
        self.guesses = []

    def step(self, action):
        assert self.action_space.contains(action)

        # Action must be a valid word
        if not tuple(action) in WORDS:
            raise InvalidWordException(encodeToStr(action) + " is not a valid word.")

        # update game board and alphabet tracking
        board_row_idx = GAME_LENGTH - self.guesses_left
        for idx, char in enumerate(action):

            if self.hidden_word[idx] == char:
                encoding = 2
            elif char in self.hidden_word:
                encoding = 1
            else:
                encoding = 0

            self.board[board_row_idx, idx] = encoding
            self.alphabet[char] = encoding

        # update guesses remaining tracker
        self.guesses_left -= 1

        # update previous guesses made
        self.guesses.append(action)

        # check to see if game is over
        if all(self.board[board_row_idx, :] == 2):
            reward = 1.0
            done = True
        else:
            if self.guesses_left > 0:
                reward = 0.0
                done = False
            else:
                reward = -1.0
                done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return {'board': self.board, 'alphabet': self.alphabet}

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        self.hidden_word = random.choice(WORDS)
        self.guesses_left = GAME_LENGTH
        self.board = np.negative(
            np.ones(shape=(GAME_LENGTH, WORD_LENGTH), dtype=int))
        self.alphabet = np.negative(np.ones(shape=(26,), dtype=int))
        self.guesses = []
        return self._get_obs()

    def render(self, mode="human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        print('###################################################')
        for i in range(len(self.guesses)):
            for j in range(WORD_LENGTH):
                letter = chr(ord('a') + self.guesses[i][j])
                if self.board[i][j] == 0:
                    print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 1:
                    print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 2:
                    print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            print()
        print()

        for i in range(len(self.alphabet)):
            letter = chr(ord('a') + i)
            if self.alphabet[i] == 0:
                print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == 1:
                print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == 2:
                print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == -1:
                print(letter + " ", end='')
        print()
        print('###################################################')
        print()
