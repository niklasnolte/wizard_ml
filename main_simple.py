# implementing a wizard game
import itertools
import random
import click
import numpy as np
from enum import Enum

from collections.abc import Iterable

# for compliance with the env
def make_arr(x):
    if not isinstance(x, Iterable):
        x = (x,)
    return np.array(x, dtype=np.int32)


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

    normal_colors = (red, blue, green, yellow)
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

    def is_trump(self, trump):
        return self.color == trump

    @property
    def colorcode(self):
        return self.color_code_map[self.color]

    def supersedes(self, other, trump):
        if other.is_wizard():
            return False
        elif self.is_wizard():
            return True
        elif self.is_joker():
            return False
        elif other.is_joker():
            return True
        elif other.is_trump(trump) and not self.is_trump(trump):
            return False
        elif self.is_trump(trump) and not other.is_trump(trump):
            return True
        elif self.color != other.color:
            return False
        else:
            return self.value > other.value

    def get_state(self):
        return make_arr((self.colorcode, self.value))


def fill_invalid(size, cards):
    l = size - len(cards)
    return {k: v for k, v in enumerate(cards + [Card.make_invalid().get_state()] * l)}


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
        # random.seed(45)
        random.shuffle(self.deck)

    def draw(self):
        return self.deck.pop(-1)


class Trick:
    def __init__(self):
        self.cards = []

    def color_to_serve(self):
        if not self.cards:
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
        return fill_invalid(game.last_round, [c.get_state() for c in self.cards])

    def determine_winner(self, trump):
        winner = 0
        for i, card in enumerate(self.cards):
            if card.supersedes(self.cards[winner], trump):
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

    def playable_cards(self, color_to_serve=None):
        my_colors = [c.color for c in self.cards]
        return [
            not color_to_serve
            or c.is_special()
            or c.color == color_to_serve
            or color_to_serve not in my_colors
            for c in self.cards
        ]

    def calc_and_set_score(self):
        if self.guessed_tricks == self.trick_count:
            self.score = 2 + self.trick_count
        else:
            self.score = -1 * abs(self.guessed_tricks - self.trick_count)

    def play_card(self, game, color_to_serve=None):
        while True:
            if self.random:
                index = random.choice(list(range(len(self.cards))))
            else:
                index = yield from game.prompt("Pick index of card to play", type=int)
            try:
                if not self.playable_cards(color_to_serve)[index]:
                    raise IndexError(f"please serve {color_to_serve}")
                return self.cards.pop(index)
            except IndexError:
                # this should not happen...
                print(f"Please pick a valid index {game.get_state_and_choice_mask()}")

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

    def get_state_and_choice_mask(self, game, color_to_serve=None):
        player_state = dict(
            score=make_arr(self.score), trick_guess=make_arr(self.guessed_tricks)
        )
        card_states = [c.get_state() for c in self.cards]

        color_mask = self.playable_cards(color_to_serve)

        # get the choice mask right
        if game.game_state == GameState.PlayingCards:
            choice_mask = color_mask + [0]*(game.last_round + 1 - len(color_mask))
        else:
            choice_mask = [1] * game.last_round + [1]

        # fill with invalid cards
        player_state["cards"] = fill_invalid(game.last_round, card_states)
        return player_state, make_arr(choice_mask)


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

    def play_round(self, Round):
        trump = None  # random.choice(Card.normal_colors + (None,))
        _print(f"\nTRUMP FOR THIS ROUND: {trump}")
        _print(f"\n\nStarting round {Round}\n")
        cards = CardStack()
        cards.shuffle()
        for _ in range(Round):
            for p in self.players:
                p.recieve_card(cards.draw())

        self.game_state = GameState.GuessingTricks
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
            winner_idx = self.current_trick.determine_winner(trump)
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
            p.calc_and_set_score()
        self.game_state = GameState.RoundFinished

    def get_state_and_choice_mask(self):
        game_state = dict()
        rotate_idx = 0
        for i, p in enumerate(self.players):
            if p.n == 0:
                rotate_idx = i
        for p in rotate(self.players, rotate_idx):
            state, choice_mask = p.get_state_and_choice_mask(
                self, self.current_trick.color_to_serve()
            )
            game_state[f"Player_{p.n}"] = state
            if not p.random:  # CAUTION works only for 1 non-random player
                player_choice_mask = make_arr(choice_mask)
        game_state["trick"] = self.current_trick.get_state(self)
        return {
            "state": game_state,
            "constraint": player_choice_mask,
            "round_done": self.game_state == GameState.RoundFinished,
            "game_over": self.game_state == GameState.Done,
        }

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
    while True:
        try:
            game.send(0)
        except StopIteration:
            game = g.play()
            next(game)
