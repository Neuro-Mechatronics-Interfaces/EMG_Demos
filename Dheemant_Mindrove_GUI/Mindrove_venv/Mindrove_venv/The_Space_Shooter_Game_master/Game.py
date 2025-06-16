import pygame as pg
from The_Space_Shooter_Game_master.Settings import *
from The_Space_Shooter_Game_master.Sprites import *
# from Settings import *
# from Sprites import *
import queue 
import shelve
from pygame.math import Vector2

PLAYER_HORIZONTAL_SPEED = 5 # Speed for player movement
class Game:
    def __init__(self, direction_queue = None):
        """            ---Game class constructor---
            It initializes the ame's major components like loading the game's sounds.
            IT initializes pygame's all instances sets the game in RUNNING STATE.
            It initializes the game screen and initializes pygame.mixer for background music.
            After successfully initializing the game to runnable state , it calls the new() method. """
        pg.init()#initializes the pygame components for display
        pg.mixer.init()#initializes pygame's audio components
        pg.mixer.music.load(path+'background.mp3')#loads background music for the game
        self.SHOOT_SOUND = pg.mixer.Sound(path+'shootLaser.wav')#loads shooting sound for te player
        self.EXPLOSION_SOUND=pg.mixer.Sound(path+'fighterKill.wav')#loads player's explosion sound
        self.EXPLOSION_MOB=pg.mixer.Sound(path+'meteorKill.wav')#loads enemies explosion sound
        pg.mixer.music.play(-1)#for looping infintely loop
        self.screen=pg.display.set_mode((WIDTH,HEIGHT))#initializing the game screen
        pg.display.set_caption('THE SPACE SHOOTER')
        # pg.display.toggle_fullscreen()
        self.font_name = pg.font.match_font('Tonto', 5)#for displaying score
        self.clock=pg.time.Clock()#for game's clock
        self.running=True#game's running state
        self.playing=False
        self.background=PGIMLOAD(path+'Starscape.png')#loads background for game
        self.GAME_START=pg.transform.scale(PGIMLOAD(path+'GAME_START.png'),(WIDTH,HEIGHT))
        self.GAME_OVER_HIGH_SCORE=pg.transform.scale(PGIMLOAD(path+'GAME_OVER_HIGH_SCORE.png'),(WIDTH,HEIGHT))
        self.GAME_OVER=pg.transform.scale(PGIMLOAD(path+'GO.png'),(WIDTH,HEIGHT))
        self.GAME_PAUSE=pg.transform.scale(PGIMLOAD(path+'GAME_PAUSE.png'),(WIDTH,HEIGHT))
        self.life=PGIMLOAD(path+'life.png')#loads life image for player's lives
        self.musicPlaying=True#sets music to running state
        self.high_score=self.load_high_score()#loads high score from previous times

        self.direction_queue = direction_queue
        # New attribute to store continuous movement from predictions
        self.predicted_player_dx = 0
        self.game_active = True # Important for prediction processing

        self.new()  # calls new method for loading new game
        

    def load_high_score(self):
        """This method loads high score from previous times using shelve api."""
        x=0
        try:
            s=shelve.open('high_score.db')#opens the record file for high score
            x=s['high_score']#gets high score if no exception is raised
            s.close()#successfully closes the record file
        except:#if any excpetion is raised
            pass
        finally:
            return x#finaaly returns the high score

    def save_high_score(self):
        """This method saves player's score as new high score when score>current high score"""
        try:
            s=shelve.open('high_score.db')#opens record file for saving the high score
            s['high_score']=self.high_score#saves the high score
            s.close()#successfully saves the high score
        except:#if any exception is raised just pass
            pass

    def new(self):
        """"  This method initializes the game's start screen and then enters into the GAME LOOP and finally calls game_over method."""
        while self.running:
            self.start_screen()#loads the start screen of the game
            self.game_loop()#game loop
            self.game_over()#handles events after game over

    def new_game(self):
        """This method initializes all sprites groups and new player's state sets game to PLAYING STATE and loads enemies and meteorites sprites."""
        self.font = pg.font.Font(self.font_name, TEXTSIZE)#object to render player's score
        self.all_sprites = pg.sprite.Group()#group of all sprites
        self.bullets = pg.sprite.Group()#group of bullets fired by the player
        self.mobs = pg.sprite.Group()#group of all meteorites and enemeies
        self.mobs_bullets = pg.sprite.Group()#group of bullets fired by the enemy ship
        self.meteorites = pg.sprite.Group()#group of all meteorites
        self.playing=True#setting game to PLAYING STATE
        self.player = Player(self.all_sprites, self.bullets, self.SHOOT_SOUND,self.EXPLOSION_SOUND)#loads the player sprite
        for i in range(1, 2):
            Mob(self.mobs, self.meteorites, self.all_sprites,self.EXPLOSION_MOB)#meteorite
        for i in range(1, 2):
            Enemy(self.mobs, self.mobs_bullets, self.all_sprites,self.EXPLOSION_MOB)#loads Enemies
        self.predicted_player_dx = 0

    def start_screen(self):
        """It handles the events occuring on the start screen of the game,asks user for new Game."""
        run=True
        while run:
            self.clock.tick(FPS)
            #Enters the event loop for handling the events
            for event in pg.event.get():
                if event.type==pg.QUIT :#if cross button is pressed ->Exits the game
                    run=False
                    self.playing=False
                    self.running=False
                if event.type==pg.KEYDOWN:
                    if event.key==pg.K_TAB:
                        run = False
                        self.playing = False
                        self.running = False

                    if event.key==pg.K_n:#if n is pressed NEW GAME is loaded by calling new_game method
                        self.new_game()
                        run=False
                        print('new game started')
                    if event.key==pg.K_ESCAPE:#if ESCAPE is pressed,->Exits the game by displaying Game over screen.
                        run=False
                        self.game_over()
            self.screen.blit(self.GAME_START, (0, 0))
            pg.display.flip()#renders the start screen on the display
        pass

    def pause(self):
        """This method handles the events occuring on the pause screen of the game."""
        run=True#sets state of pause screen as running
        while run:#enters the pause screen loop
            self.clock.tick(FPS)
            #Event loop for the game pause screen
            for event in pg.event.get():
                if event.type==pg.QUIT :#if Cross button is pressed -> Exits the game
                    self.playing = False
                    self.running = False
                    run = False
                if event.type==pg.KEYDOWN:
                    if event.key==pg.K_TAB:
                        self.playing = False
                        self.running = False
                        run = False
                    elif event.key==pg.K_x:#if x is pressed -> Exits the game
                        self.playing = False
                        self.running = False
                        run = False
                    elif event.key==pg.K_ESCAPE:#if ESCAPE is pressed -> Game is resumed
                        run = False
                    elif event.key==pg.K_n:#if n is pressed -> new game is loaded
                        self.new()#loads the new game
                        run = False
                    elif event.key==pg.K_m:#if m is pressed -> music is paused or resumed
                        if self.musicPlaying:#if usic is currently playing then
                            self.musicPlaying=False
                            print('music stopped')
                            pg.mixer.music.pause()#pause the background music
                        else:#if music is already paused then
                            print('music resumed')
                            self.musicPlaying=True
                            pg.mixer.music.unpause()#music is resumed
            self.screen.blit(self.GAME_PAUSE,(0,0))#blits the game pause screen
            pg.display.flip()#renders the pause screen to the display

    def game_over(self):
        """It handles events that occur on game over screen."""
        self.now=pg.time.get_ticks()
        pg.mixer.music.pause()
        while pg.time.get_ticks()-self.now<3000:
            self.clock.tick(FPS)
            pass

        run=True#sets game over screen to running state
        high=False#state for high score status
        if self.player.score>self.high_score:#if current score is greater than current high score
            self.high_score=self.player.score#sets new high score as current score
            font=pg.font.Font(self.font_name,100)#objec to render new high score
            text=font.render(str(self.high_score),True,(255,255,255))#surface showing new high score
            self.save_high_score()#saves new high score by calling save_high-score() method
            high=True#sets new high score state
        while run:
            self.clock.tick(FPS)
            #enters game over's event loop
            for event in pg.event.get():
                if event.type==pg.QUIT :#if cross button is pressed->exit the game
                    run=False
                    self.running=False
                if event.type==pg.KEYDOWN:
                    if event.key==pg.K_TAB:
                        run = False
                        self.running = False
                    if event.key==pg.K_n:#if n is pressed -> go to start screen
                        run=False
                        pass
            if high:#checks new high score state
                self.screen.blit(self.GAME_OVER_HIGH_SCORE, (0, 0))#blits new high score game over screen
                self.screen.blit(text,(250,350))#blits new high score
            else:#otherwise
                self.screen.blit(self.GAME_OVER,(0,0))#blits game over screen
            pg.display.flip()#renders to the screen

    def drawScore(self,score):
        """This method displays the user's current score on the game screen."""
        text=self.font.render(score,True,TEXTCOLOR)#returns text surface to show current score
        self.screen.blit(text,(WIDTH/2,10))#renders text surface to the display

    def drawHealth(self,health):
        """This method displays user's health on the game screen."""
        pg.draw.rect(self.screen,pg.color.Color('WHITE'),(10,10,WIDTH/2-20,30),2)
        pg.draw.rect(self.screen,pg.color.Color('GREEN'),(15,15,health*(WIDTH/2-30)/100,20))

    def drawLife(self,life):
        """This method displays player's lives on the game screen."""
        x = WIDTH - 51
        for i in range(life):
            self.screen.blit(self.life, (x, 10))
            x -= 51

    def game_loop(self):
        """MAIN GAME LOOP--> TAKE EVENTS-> UPDATE EVERYTHING -> DRAW ON THE SCREEN."""
        while self.playing:#enters the game loop
            self.clock.tick(FPS)
            self.events()#takes events from the player for movements and shoot
            self.update()#updates everything i.e., all the sprites on the screen
            self.draw()#renders everything on the game screen

    def update_direction_from_prediction(self, predicted_label):
        """
        Updates the predicted player movement direction or triggers an action
        based on the incoming predicted label.
        """
        if not self.game_active:
            return

        label = predicted_label.lower()
        if label == 'findex':
            self.player.shoot()  # Calls player class shoot method
            self.predicted_player_dx = 0 # Stop movement after shooting if not explicitly moving
        elif label == 'wflex':
            self.predicted_player_dx = -PLAYER_HORIZONTAL_SPEED  # Move left
        elif label == 'wexten':
            self.predicted_player_dx = PLAYER_HORIZONTAL_SPEED  # Move right
        elif label == 'rest': # New condition for stopping
            self.predicted_player_dx = 0  # Stop movement

    def events(self):
        """This method takes events from the user."""
        # Initialize keyboard-based movement for the current frame
        dx_keyboard = 0

        # Process predicted labels from the queue first
        if self.direction_queue and not self.direction_queue.empty():
            try:
                predicted_label = self.direction_queue.get_nowait()
                print(f"[Game] Received prediction: {predicted_label}")
                self.update_direction_from_prediction(predicted_label)
            except queue.Empty:
                pass  # Queue was empty, no new prediction
            except Exception as e:
                print(f"Queue read error: {e}")

        # Process Pygame events (keyboard presses, window close, etc.)
        for event in pg.event.get():
            if event.type == pg.QUIT:  # if cross button is pressed -> GAME OVER
                self.playing = False
                self.running = False  # Ensure main game loop exits
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_TAB:
                    self.playing = False
                    self.running = False  # Ensure main game loop exits
                if event.key == pg.K_ESCAPE:  # if ESCAPE is pressed -> GAME PAUSED
                    self.pause()  # calls pause() method

        # Get current state of all keys for continuous keyboard movement
        keys = pg.key.get_pressed()

        if keys[pg.K_LEFT]:  # if LEFT arrow is pressed -> move left
            dx_keyboard -= PLAYER_HORIZONTAL_SPEED
        if keys[pg.K_RIGHT]:  # if RIGHT arrow is pressed -> move right
            dx_keyboard += PLAYER_HORIZONTAL_SPEED
        if keys[pg.K_SPACE]:  # if SPACE is pressed -> shoot
            self.player.shoot()  # calls player class shoot method

        # Determine the final movement for the player
        # If keyboard input is present, it overrides predicted movement for that axis,
        # or you can combine them (e.g., dx_keyboard + self.predicted_player_dx)
        # For this design, let's prioritize keyboard if pressed, otherwise use prediction
        final_dx = dx_keyboard if dx_keyboard != 0 else self.predicted_player_dx

        # Apply the final movement to the player
        self.player.move(final_dx)  # calls move method of player object to move by dx units across X-axis

    def update(self):
        """This method updates every sprites new positions and other animations by calling update method of Sprite groups."""
        hits=pg.sprite.spritecollide(self.player,self.mobs,True)#if player is hit by an enemy or meteorite
        for hit in hits:
            self.player.health-=hit.stealth#reduces the health of player
            self.player.isDamaged=True#sets player to damage state
            # if hit.name=='meteor':#checks if player hit the meteorite
            #     Mob(self.mobs,self.meteorites,self.all_sprites,self.EXPLOSION_MOB)#a new meteorite come on the screen
            # else:#if player hit the enemy
            #     Enemy(self.mobs,self.mobs_bullets,self.all_sprites,self.EXPLOSION_MOB)#a new enemy ship comes

        hits=pg.sprite.groupcollide(self.mobs,self.bullets,False,True)#if palyer's bullets hit the enemies or meteorites
        for hit in hits:
            hit.health-=40#reduces health of meteorite or enemy
            hit.stealth-=20#reduces enemies or meteorite's stealth by 5 units
            if hit.health<=0:#if enemy's or meteoite's health <=0 kill the enemy or meteorite
                hit.kill()#kills
                if hit.name == 'meteor':#if hit a meteorite
                    Mob(self.mobs,self.meteorites,self.all_sprites,self.EXPLOSION_MOB)#new meteorite comes
                    Enemy(self.mobs, self.mobs_bullets, self.all_sprites,self.EXPLOSION_MOB)
                    if hit.isBig:#if hit to big meteorite
                        self.player.score+=40#increase player's score by 40 units
                    else:#if hit a small meteorite
                        self.player.score+=20#increase player's score by 20 units
                else:#or hit an enemy ship or UFO
                    Enemy(self.mobs, self.mobs_bullets, self.all_sprites,self.EXPLOSION_MOB)#new enemy comes
                    self.player.score+=20#increase player's score by 10 units
            if 40 <= self.player.score <= 50:
                Enemy(self.mobs, self.mobs_bullets, self.all_sprites,self.EXPLOSION_MOB)#new enemy comes
                Mob(self.mobs,self.meteorites,self.all_sprites,self.EXPLOSION_MOB)

            if 100 <= self.player.score <= 150:
                Enemy(self.mobs, self.mobs_bullets, self.all_sprites,self.EXPLOSION_MOB)#new enemy comes
                Mob(self.mobs,self.meteorites,self.all_sprites,self.EXPLOSION_MOB)

        hits=pg.sprite.spritecollide(self.player,self.mobs_bullets,True)#if player is hit with enemy's bullets
        for hit in hits:
            self.player.health-=4#reduces player's health by 4 units
            self.player.isDamaged=True#sets player to damaged state

        hits=pg.sprite.groupcollide(self.mobs_bullets,self.meteorites,True,False)#if enemy's bullets hit a meteorite
        for hit in hits:
            for h in hits[hit]:
                h.health-=4#reduces meteorite's health by 4 units
                if h.health<=0:#if meteorite's health is zero
                    h.kill()#kill the meteorite
                    # Mob(self.mobs, self.meteorites, self.all_sprites, self.EXPLOSION_MOB)  # new meteorite comes
        self.all_sprites.update()#update position of all sprites on the screen

        if self.player.lives<0:#if player has no life left then -> GAME OVER
            self.playing=False
        pass

    def draw(self):
        """This method draws every sprite on the screen by calling SPRITE groups's draw() method"""
        self.screen.blit(self.background,(0,0))#draws game's background screen
        self.all_sprites.draw(self.screen)#draws all sprites on the screen
        self.drawScore(str(abs(self.player.score)))#draws palyer's score on the screen
        self.drawHealth(self.player.health)#draws player's health on the game screen
        self.drawLife(self.player.lives)#draws player's no. of lives on the screen
        pg.display.flip()#finally renders the screen
        pass

# if __name__=='__main__':
#     Game()
