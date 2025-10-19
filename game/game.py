import random
from collections import namedtuple
import pygame

# Initialize pygame display for image loading
pygame.display.init()
pygame.font.init() # Ensure font is initialized for any text rendering if needed

Card = namedtuple('Card', ['rank', 'suit'])
TARGET_SCORE = 50

class Deck:
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'] # Changed to '1' for Ace
    suits = ['spades', 'hearts', 'diamonds', 'clubs'] # Reordered to match PettingZoo
    
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in self.suits
                      for rank in self.ranks]

    def __len__(self):
        return len(self.cards)

    def remove(self, card):
        """
        Removes given card from deck
        :param card: card object
        """
        if card in self.cards:
            self.cards.remove(card)

    def random_choice(self):
        """
        Selects a card at random from the card deck, returns that card and removes that card from the deck
        :return: card object
        """
        if not self.cards:
            return None # Handle empty deck
        card = random.choice(self.cards)
        self.remove(card)
        return card

    @staticmethod
    def get_image(card):
        """
        Takes a card object and returns the corresponding image of that card
        :param card: card object (tup)
        :return: pygame surface
        """
        try:
            # Use f-strings for cleaner path
            image = pygame.transform.scale(pygame.image.load(f"game/deck_images/{card.rank}_of_{card.suit}.png"), (70, 100))
            return image
        except pygame.error:
            # Fallback for missing images
            print(f"Warning: Could not load image deck_images/{card.rank}_of_{card.suit}.png")
            fallback = pygame.Surface((70, 100))
            fallback.fill((255, 255, 255))
            font = pygame.font.SysFont("comicsans", 12)
            text = font.render(f"{card.rank} {card.suit}", True, (0,0,0))
            fallback.blit(text, (5, 5))
            return fallback

class Game(Deck):
    def __init__(self, gid):
        super().__init__()
        self.gid = gid
        self.ready = False
        self_card = self.random_choice()
        if self_card:
            self.middle_card = self_card
        else:
            self.middle_card = Card('1', 'spades') # Fallback
            
        self.target_score = TARGET_SCORE
        self.p_scores = [0, 0]
        self.wins = [0, 0]
        self.active_p = random.choice([0, 1])
        self.turn = 0
        
        # Simplified state for wrapper
        self.win_round = {"win": False, "type": "", "player": None}
        
        # Removed network-specific attributes
        # self.end_hands = {0: {"sets": None, "leftovers": None}, 1: {"sets": None, "leftovers": None}}
        # self.deadwoods = {0: None, 1: None}
        # self.ready_continue = [False, False]
        # self.player_reset = [False, False]
        # self.new_hands = []
        # self.ready_new_game = [False, False]

    def next_turn(self):
        self.turn += 1

    def connected(self):
        """
        Returns whether the client is connected or not
        :return: bool
        """
        return self.ready