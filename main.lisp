;A Poker Texas Hold`Em engine in Common Lisp

;Copyright: Paulo Garcia (paulo.g@chula.ac.th)
;International School of Engineering, 
;Faculty of Engineering,
;Chulalongkorn University, Thailand

;Licensed under CC -BY-NC-SA 4.0
;(Creative Commons Attribution-NonCommercial-ShareAlike)

(require "Players" "players.lisp")
(require "Poker" "poker.lisp")

;initializes a game state, given number of players
; default, each player gets 1000 chips
(defun init-state (players)
	(make-list (list-length players) :initial-element 1000)
)

;tests a single round, using only the 3 default noob players
(defun test-1-round ()
	(game-round 
		;initial game state, for 3 players
		(list 
			(list `noob1 1000 0 `in)
			(list `noob2 1000 0 `in)
			(list `noob3 1000 0 `in)
		)
		;list of players "set-hand" function to deal cards 
		(list `noob1-set-hand `noob2-set-hand `noob3-set-hand)
		;list of players "get-hand" function to determine winner
		(list `noob1-get-hand `noob2-get-hand `noob3-get-hand)
	)
)

;tests a full game, using only the 3 default noob players
(defun test-full ()
	(full-game 
		;initial game state, for 3 players
		(list 
			(list `noob1 1000 0 `in)
			(list `noob2 1000 0 `in)
			(list `noob3 1000 0 `in)
		)
		;list of players "set-hand" function to deal cards 
		(list `noob1-set-hand `noob2-set-hand `noob3-set-hand)
		;list of players "get-hand" function to determine winner
		(list `noob1-get-hand `noob2-get-hand `noob3-get-hand)
	)
)

;tests a full game, using the 3 default noob players
;and your AI
(defun test-full-AI ()
	(full-game 
		;initial game state, for 4 players
		(list 
			(list `noob1 1000 0 `in)
			(list `noob2 1000 0 `in)
			(list `noob3 1000 0 `in)
      (list `AI    1000 0 `in)
		)
		;list of players "set-hand" function to deal cards 
		(list 
      `noob1-set-hand `noob2-set-hand 
      `noob3-set-hand `AI-set-hand
    )
		;list of players "get-hand" function to determine winner
		(list 
      `noob1-get-hand `noob2-get-hand 
       `noob3-get-hand `AI-get-hand
    )
	)
)

