# implementing a wizard game
import itertools
import random
import click

trump = None

def rotate(l, n):
    return l[n:] + l[:n]

class Card:
    possible_colors = ('red', 'blue', 'green', 'yellow')
    possible_values = range(1, 14)
    joker_color = wizard_color = 'white'
    joker_value = 0
    wizard_value = 14

    def __init__(self, color, value):
        """
        defining a wizard card
        """
        if value in (self.joker_value, self.wizard_value):
            assert(color == self.joker_color)
        self.color = color
        self.value = value

    def __repr__(self):
        return f"({self.color}, {self.value})"

    def is_wizard(self):
        return self.value == self.wizard_value

    def is_joker(self):
        return self.value == self.joker_value

    def is_trump(self):
        global trump
        return self.color == trump

    def __gt__(self, other):
        global trump
        if other.is_wizard():
            return False
        elif self.is_wizard():
            return True;
        elif other.is_trump() and not self.is_trump():
            return False
        elif self.is_trump() and not other.is_trump():
            return True
        elif self.color != other.color:
            return False
        else:
            return self.value > other.value

class card_stack:
    all_cards = [Card(color, value) for value, color in
                 list(itertools.product(Card.possible_values, Card.possible_colors)) +
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


class Player:
    def __init__(self, number):
        self.n = number
        self.score = 0
        self.cards = []
        self.guessed_tricks = -1
        self.trick_count = 0

    def recieve_card(self, card):
        self.cards.append(card)

    def play_card(self):
        while True:
            index = click.prompt("Pick index of card to play", type=int)
            try:
                return self.cards.pop(index)
            except IndexError:
                click.echo("Please pick a valid index")

    def show_cards_with_index(self):
        click.echo(f"\n\nCards of Player {self.n}:")
        for i, card in enumerate(self.cards):
            click.echo(f"[{i}] : {card}")
        click.echo()

    def guess_tricks(self):
        self.guessed_tricks = click.prompt(f"Player {self.n}, how many tricks will you get?", type=int)

    def reset_tricks(self):
        self.guessed_tricks = -1
        self.trick_count = 0

class Game:
    def __init__(self, nplayers):
        self.nplayers = nplayers
        self.players = [Player(i) for i in range(nplayers)]
        self.last_round = int(60/nplayers)

    def play(self):
        click.echo("Lets go!")

        for r in range(self.last_round):
            self.play_round(r+1)
            click.echo("Scores after round {}:\n".format(r+1))
            for p in self.players:
                click.echo(f"Player {p.n}: {p.score}")
                p.reset_tricks()
            click.echo("")
            self.players = rotate(self.players, 1)


        winner = 0
        score = self.players[0].score
        for p in self.players[1:]:
            if p.score > score:
                winner = p.n
        click.echo(f"Congratz to Player {winner}, who won with a score of {score}")

    @staticmethod
    def determine_score(guess, count):
        if guess == count:
            return 2 + guess
        else:
            return -1*abs(guess-count)

    def play_round(self, Round):
        global trump
        trump = random.choice(card.possible_colors + (None,))
        print(f"\n\nStarting round {Round}\n")
        cards = card_stack()
        cards.shuffle()
        for _ in range(Round):
            for p in self.players:
                p.recieve_card(cards.draw())

        for p in self.players:
            p.show_cards_with_index()
            p.guess_tricks()

        winner_idx = 0 # not really, only the guy who starts
        players = list(self.players) # this list will be rotated to change the starting player
        for _ in range(Round): #there are #rounds cards per player
            current_trick = []
            players = rotate(players, winner_idx)
            for p in players:
                p.show_cards_with_index()
                current_trick.append(p.play_card())
            winner_idx = self.determine_winner(current_trick)
            winner = players[winner_idx]
            winner.trick_count += 1
            click.echo(f"Player {winner.n} won this trick.\n\n")
            for p in players:
                click.echo(f"Player {p.n}, you now have {p.trick_count} tricks, and you need {p.guessed_tricks}")
            click.echo("")

        for p in self.players:
            p.score += self.determine_score(p.guessed_tricks, p.trick_count)


    @staticmethod
    def determine_winner(cards):
        winner = 0
        color_to_be_played = cards[0].color
        for i,card in enumerate(cards):
            if card > cards[winner]:
                winner = i
        return winner

g = Game(click.prompt("Number of players?", type = int))
g.play()
