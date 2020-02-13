# implementing a wizard game
import itertools
import random
import click
import numpy as np
from enum import Enum

trump = None

class GameState(Enum):
    GuessingTricks = 1
    PlayingCards = 2
    RoundFinished = 3
    Done = 4


def rotate(l, n):
    return l[n:] + l[:n]


class Card:
    white = "white"
    red = "red"
    blue = "blue"
    green = "green"
    yellow = "yellow"
    invalid_color = "INVALID"
    color_code_map = {
        y: x for x, y in enumerate([invalid_color, white, red, blue, green, yellow])
    }

    normal_colors = (red,)  # blue, green, yellow)
    normal_values = list(range(1, 14))
    joker_color = wizard_color = white
    joker_value = 0
    wizard_value = 14

    def is_special(self):
        return self.color == "white"

    @classmethod
    def make_invalid(cls):
        return cls(cls.invalid_color, -1)

    def is_invalid(self):
        return self.color == self.invalid_color

    def __init__(self, color, value):
        """
        defining a wizard card
        """
        if value in (self.joker_value, self.wizard_value):
            assert color == self.joker_color
        self.color = color
        self.value = value

    def __repr__(self):
        return f"Card({self.color}, {self.value})"

    def is_wizard(self):
        return self.value == self.wizard_value

    def is_joker(self):
        return self.value == self.joker_value

    def is_trump(self):
        global trump
        return self.color == trump

    @property
    def colorcode(self):
        return self.color_code_map[self.color]

    def __gt__(self, other):
        global trump
        if other.is_wizard():
            return False
        elif self.is_wizard():
            return True
        elif self.is_joker():
            return False
        elif other.is_trump() and not self.is_trump():
            return False
        elif self.is_trump() and not other.is_trump():
            return True
        elif self.color != other.color:
            return False
        else:
            return self.value > other.value

    def get_state(self):
        return (self.colorcode, self.value)


class CardStack:
    all_cards = [
        Card(color, value)
        for value, color in list(
            itertools.product(Card.normal_values, Card.normal_colors)
        )
        + 4
        * [(Card.joker_value, Card.joker_color), (Card.wizard_value, Card.wizard_color)]
    ]

    def __init__(self):
        """
        all wizard cards in a stack
        """
        self.deck = list(self.all_cards)

    def shuffle(self):
        """
        shuffle the card deck
        """
        # random.seed(43)
        random.seed(45)
        random.shuffle(self.deck)

    def draw(self):
        return self.deck.pop(-1)


class Trick:
    def __init__(self):
        self.cards = []

    def color_to_serve(self):
        if not len(self.cards):
            return None
        for card in self.cards:
            if card.is_wizard():
                return None
            if card.is_joker():
                continue
            return card.color

    def add(self, card):
        self.cards.append(card)

    def __str__(self):
        return str(self.cards)

    def get_state(self, game):
        state = []
        for i in range(game.last_round):
            try:
                state.append(self.cards[i].get_state())
            except IndexError:
                state.append(Card.make_invalid().get_state())
        return state

    def determine_winner(self):
        winner = 0
        for i, card in enumerate(self.cards):
            if card > self.cards[winner]:
                winner = i
        return winner


class Player:
    def __init__(
        self, number, random=False
    ):  # implement a random player to play against the AI
        self.n = number
        self.score = 0
        self.cards = []
        self.guessed_tricks = -1
        self.trick_count = 0
        self.random = random

    def recieve_card(self, card):
        self.cards.append(card)

    def play_card(self, game, color_to_serve=None):
        while True:
            if self.random:
                index = random.choice(list(range(len(self.cards))))
            else:
                index = yield from game.prompt("Pick index of card to play", type=int)
            try:
                if (
                    color_to_serve
                    and not self.cards[index].is_special()
                    and self.cards[index].color != color_to_serve
                    and color_to_serve in [card.color for card in self.cards]
                ):
                    raise IndexError(f"please serve {color_to_serve}")
                return self.cards.pop(index)
            except IndexError:
                _print("Please pick a valid index")

    def show_cards_with_index(self):
        _print(f"\n\nCards of Player {self.n}:")
        for i, card in enumerate(self.cards):
            _print(f"[{i}] : {card}")
        _print()

    def guess_tricks(self, game):
        if self.random:
            self.guessed_tricks = random.choice(list(range(len(self.cards))))
        else:
            self.guessed_tricks = yield from game.prompt(
                f"Player {self.n}, how many tricks will you get?", type=int
            )
        _print(f"Player {self.n} called {self.guessed_tricks} tricks!\n")

    def reset_tricks(self):
        self.guessed_tricks = -1
        self.trick_count = 0

    def __repr__(self):
        return f"Player(n={self.n},score={self.score})"

    def get_state_and_choice_mask(self, game):
        player_state = [self.score, self.guessed_tricks]
        card_states = [c.get_state() for c in self.cards]

        # get the choice mask right
        if game.game_state == GameState.PlayingCards:
            choice_mask = [i < len(card_states) for i in range(game.last_round)] + [0]
        else:
            choice_mask = [1]*game.last_round + [1]

        #fill with invalid cards
        card_states.extend([Card.make_invalid().get_state()]*(game.last_round - len(card_states)))
        
        return player_state+card_states, choice_mask


class Game:
    def __init__(self, nplayers, random_idxs=[0], print_function=print):
        self.nplayers = nplayers
        self.last_round = 2
        self.players = [Player(i, i in random_idxs) for i in range(nplayers)]
        self._random_idxs = random_idxs
        self.current_trick = Trick()
        self.game_state = GameState.GuessingTricks
        global _print
        _print = print_function

    def play(self):
        _print("Lets go!")

        for r in range(1, self.last_round):
            yield from self.play_round(r + 1)
            _print("Scores after round {}:\n".format(r + 1))
            for p in self.players:
                _print(f"Player {p.n}: {p.score}")
                p.reset_tricks()
            _print("")
            self.players = rotate(self.players, 1)  # the next one starts

        winner = 0
        score = -1000  # small number
        for p in self.players:
            if p.score > score:
                winner = p.n
                score = p.score
        _print(f"Congratz to Player {winner}, who won with a score of {score}")
        self.game_state = GameState.Done
        yield self.get_state_and_choice_mask()

    @staticmethod
    def determine_score(guess, count):
        if guess == count:
            return 2 + guess
        else:
            return -1 * abs(guess - count)

    def play_round(self, Round):
        global trump
        trump = None  # random.choice(Card.normal_colors + (None,))
        _print(f"\nTRUMP FOR THIS ROUND: {trump}")
        _print(f"\n\nStarting round {Round}\n")
        cards = CardStack()
        cards.shuffle()
        for _ in range(Round):
            for p in self.players:
                p.recieve_card(cards.draw())

        self.game_state = GameState.GuessingTricks
        yield self.get_state_and_choice_mask()
        for p in self.players:
            p.show_cards_with_index()
            yield from p.guess_tricks(self)

        winner_idx = 0  # not really, only the guy who starts
        self.game_state = GameState.PlayingCards
        for _ in range(Round):  # there are #rounds cards per player
            self.current_trick = Trick()
            players = rotate(self.players, winner_idx)
            for p in players:
                p.show_cards_with_index()
                color_to_serve = self.current_trick.color_to_serve()
                card = yield from p.play_card(self, color_to_serve)
                self.current_trick.add(card)
                _print(f"\nCurrent trick: {self.current_trick}\n")
            winner_idx = self.current_trick.determine_winner()
            winner = players[winner_idx]
            winner.trick_count += 1
            _print(f"Player {winner.n} won this trick.\n\n")
            for p in players:
                _print(
                    f"Player {p.n}, you now have {p.trick_count} tricks, and you need {p.guessed_tricks}"
                )
            _print("")
            _print(f"\nState: {self.get_state_and_choice_mask()}\n")

        for p in self.players:
            p.score += self.determine_score(p.guessed_tricks, p.trick_count)
        self.game_state = GameState.RoundFinished

    def get_state_and_choice_mask(self):
        game_state = []
        rotate_idx = 0
        for i, p in enumerate(self.players):
            if p.n == 0:
                rotate_idx = i
        for p in rotate(self.players, rotate_idx):
            state, choice_mask = p.get_state_and_choice_mask(self)
            game_state.extend(state)
            if p.n not in self._random_idxs: # CAUTION works only for 1 player
                player_choice_mask = choice_mask
        game_state.extend(self.current_trick.get_state(self))
        game_state.append(self.game_state == GameState.RoundFinished)
        game_state.append(self.game_state == GameState.Done)
        return {"state" : game_state, "mask" : player_choice_mask}

    def prompt(self, msg, type=int):
        _print(msg)
        while True:
            next_action = yield self.get_state_and_choice_mask()
            if isinstance(next_action, type):
                break
            _print(f"please provide a valid input: {type}")
        _print(f"action: {next_action}")
        return next_action


if __name__ == "__main__":
    g = Game(click.prompt("Number of players?", type=int))
    game = g.play()
    next(game)