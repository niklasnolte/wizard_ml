# implementing a wizard game
import itertools
import random
import click
import numpy as np


trump = None

def rotate(l, n):
    return l[n:] + l[:n]

class Card:
    white = 'white'
    red = 'red'
    blue = 'blue'
    green = 'green'
    yellow = 'yellow'
    invalid_color = 'INVALID'
    color_code_map = {y:x for x,y in enumerate([invalid_color, white, red, blue, green, yellow])}

    normal_colors = (red, blue, green, yellow)
    normal_values = list(range(1, 14))
    joker_color = wizard_color = white
    joker_value = 0
    wizard_value = 14

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
            assert(color == self.joker_color)
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
            return True;
        elif self.is_joker() and not other.is_joker:
            return False
        elif other.is_joker():
            return True
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
    all_cards = [Card(color, value) for value, color in
                 list(itertools.product(Card.normal_values, Card.normal_colors)) +
                 4*[(Card.joker_value, Card.joker_color), (Card.wizard_value, Card.wizard_color)] ]

    def __init__(self):
        """
        all wizard cards in a stack
        """
        self.deck = list(self.all_cards)

    def shuffle(self):
        """
        shuffle the card deck
        """
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
        color_to_be_played = self.color_to_serve()
        for i,card in enumerate(self.cards):
            if card > self.cards[winner]:
                winner = i
        return winner

class Player:
    def __init__(self, number, random=False): #implement a random player to play against the AI
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
                index = game.prompt("Pick index of card to play", type=int)
            try:
                if color_to_serve and \
                   self.cards[index].color != color_to_serve and \
                   color_to_serve in [card.color for card in self.cards]:
                    raise IndexError(f"please serve {color_to_serve}")
                return self.cards.pop(index)
            except IndexError:
                click.echo("Please pick a valid index")

    def show_cards_with_index(self):
        click.echo(f"\n\nCards of Player {self.n}:")
        for i, card in enumerate(self.cards):
            click.echo(f"[{i}] : {card}")
        click.echo()

    def guess_tricks(self, game):
        if self.random:
            self.guessed_tricks = random.choice(list(range(len(self.cards))))
        else:
            self.guessed_tricks = game.prompt(f"Player {self.n}, how many tricks will you get?", type=int)
        click.echo(f"Player {self.n} called {self.guessed_tricks} tricks!\n")

    def reset_tricks(self):
        self.guessed_tricks = -1
        self.trick_count = 0

    def __repr__(self):
        return f"Player(n={self.n},score={self.score})"

    def get_state(self, game):
        state = [self.score, self.guessed_tricks]
        for i in range(game.last_round):
            try:
                state.append(self.cards[i].get_state())
            except IndexError:
                state.append(Card.make_invalid().get_state())
        return state

class Game:
    def __init__(self, nplayers, random_idxs=[0], pipe = None):
        # if there is a pipe, we'll be getting instructions from somewhere else
        self.nplayers = nplayers
        self.last_round = 2
        self.players = [Player(i, i in random_idxs) for i in range(nplayers)]
        cards = CardStack()
        cards.shuffle()
        self.current_trick = Trick()
        self.pipe = pipe
        self.round_finished = False
        self.game_over = False

    def play(self):
        click.echo("Lets go!")

        for r in range(1,self.last_round):
            self.play_round(r+1)
            click.echo("Scores after round {}:\n".format(r+1))
            for p in self.players:
                click.echo(f"Player {p.n}: {p.score}")
                p.reset_tricks()
            click.echo("")
            self.players = rotate(self.players, 1) # the next one starts


        winner = 0
        score = -1000 #small number
        for p in self.players:
            if p.score > score:
                winner = p.n
                score = p.score
        click.echo(f"Congratz to Player {winner}, who won with a score of {score}")
        if self.pipe:
            self.game_over = True
            self.pipe.send(self.get_state())

    @staticmethod
    def determine_score(guess, count):
        if guess == count:
            return 2 + guess
        else:
            return -1*abs(guess-count)

    def play_round(self, Round):
        global trump
        trump = None#random.choice(Card.normal_colors + (None,))
        click.echo(f"\nTRUMP FOR THIS ROUND: {trump}")
        print(f"\n\nStarting round {Round}\n")
        cards = CardStack()
        cards.shuffle()
        for _ in range(Round):
            for p in self.players:
                p.recieve_card(cards.draw())

        for p in self.players:
            p.show_cards_with_index()
            p.guess_tricks(self)
        self.round_finished = False # new round begins

        winner_idx = 0 # not really, only the guy who starts
        players = list(self.players) # this list will be rotated to change the starting player
        for _ in range(Round): #there are #rounds cards per player
            self.current_trick = Trick()
            players = rotate(players, winner_idx)
            for p in players:
                p.show_cards_with_index()
                color_to_serve = self.current_trick.color_to_serve();
                self.current_trick.add(p.play_card(self, color_to_serve))
                click.echo(f"\nCurrent trick: {self.current_trick}\n")
            winner_idx = self.current_trick.determine_winner()
            winner = players[winner_idx]
            winner.trick_count += 1
            click.echo(f"Player {winner.n} won this trick.\n\n")
            for p in players:
                click.echo(f"Player {p.n}, you now have {p.trick_count} tricks, and you need {p.guessed_tricks}")
            click.echo("")
            click.echo(f"\nState: {self.get_state()}\n")

        for p in self.players:
            p.score += self.determine_score(p.guessed_tricks, p.trick_count)
        self.round_finished = True # round ends


    def get_state(self):
        state = []
        rotate_idx = 0
        for i,p in enumerate(self.players):
            if p.n == 0:
                rotate_idx = i
        for p in rotate(self.players, rotate_idx):
            state.extend(p.get_state(self))
        state.extend(self.current_trick.get_state(self))
        state.append(self.round_finished)
        state.append(self.game_over)
        return state

    def prompt(self, msg, type=int, **kwargs):
        if not self.pipe:
            while True:
                try:
                    return type(click.prompt(msg, type))
                except TypeError:
                    print("please give a valid input")
        else:
            self.pipe.send(self.get_state(**kwargs))
            print(msg)
            next_action = self.pipe.recv()
            print(f"action: {next_action}")
            return next_action

# g = Game(click.prompt("Number of players?", type = int))
# g.play()
