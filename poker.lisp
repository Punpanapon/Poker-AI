;A Poker Texas Hold`Em engine in Common Lisp

;Copyright: Paulo Garcia (paulo.g@chula.ac.th)
;International School of Engineering, 
;Faculty of Engineering,
;Chulalongkorn University, Thailand

;Licensed under CC -BY-NC-SA 4.0
;(Creative Commons Attribution-NonCommercial-ShareAlike)



;game engine functions 
; NOT to be called from your code
(defun init-suit (suit)
	(list 
		(list 1 suit)
		(list 2 suit)	
		(list 3 suit)
		(list 4 suit)
		(list 5 suit)
		(list 6 suit)
		(list 7 suit)
		(list 8 suit)
		(list 9 suit)
		(list 11 suit)
		(list 11 suit)
		(list 12 suit)
		(list 13 suit)
	)
)

(let
	(
		(deck nil	)
	)
	(defun see-deck () deck)
	(defun init-deck ()
		(setf deck
			(append
				(init-suit `H)
				(init-suit `D)
				(init-suit `C)
				(init-suit `S)
			)
		)
	)
	(defun get-hand ()
		(let
			(
				(hand nil)
				(n nil)
			)
			(setf n (random (list-length deck)))
			(setf hand
				(list (nth n deck))
			)
			(setf deck (remove (nth 0 hand) deck))
			(setf n (random (list-length deck)))
			(setf hand
				(append hand (list (nth n deck)))
			)
			(setf deck (remove (nth 1 hand) deck))
			hand
		)
	)
	(defun get-flop ()
		(let 
			(
				(flop nil)
				(n nil)
			)
			(setf n (random (list-length deck)))
			(setf flop 
				(list (nth n deck))
			)
			(setf deck (remove (nth 0 flop) deck))
			(setf n (random (list-length deck)))
			(setf flop
				(append flop (list (nth n deck)))
			)
			(setf deck (remove (nth 1 flop) deck))
			(setf n (random (list-length deck)))
			(setf flop 
				(append flop (list (nth n deck)))
			)
			(setf deck (remove (nth 2 flop) deck))
			flop
		)
	)
	(defun get-turn-or-river ()
		(let
				(
					(n (random (list-length deck)))
					(card nil)
				)
				(setf card (nth n deck))
				(setf deck (remove card deck))
				card		
		)
	)
)

(defun get-card-symbol (n)
	(nth (- n 1) (list #\A 2 3 4 5 6 7 8 9 10 #\J #\Q #\K))
)

(defun get-card-suit (n)
	(cond
		((equal n `D)
			#\U+2662
		)
		((equal n `H)
			#\U+2661
		)
		((equal n `S)
			#\U+2664
		)
		((equal n `C)
			#\U+2667
		)
		(t nil)
	)
)
(defun print-cards (cards)
	(dolist (card cards)
		(format t "~a" 
			(get-card-symbol (nth 0 card))
		;	(get-card-suit (nth 1 card))
		)
		(cond
			((equal (nth 1 card) `D)
				(format t (concatenate 'string "~c[" "91m" "~c " )  #\ESC #\U+2662)
				(format t (concatenate 'string "~c[" "0m" )  #\ESC)
			)
			((equal (nth 1 card) `H)
				(format t (concatenate 'string "~c[" "91m" "~c " )  #\ESC #\U+2661)
				(format t (concatenate 'string "~c[" "0m" )  #\ESC)
			)
			((equal (nth 1 card) `S)
				(format t "~c " 	#\U+2664)
			)
			((equal (nth 1 card) `C)
				(format t "~c " 	#\U+2667)
			)
			(t nil)
		)
	)
	(format t "~%")
)

(defun count-cards (cards card)
	(list-length 
		(remove-if
			(lambda (x)
				(if (equal (nth 0 x) card)
					nil
					t
				)
			)
			cards
		)
	)
)
(defun count-suit (cards suit)
	(list-length 
		(remove-if
			(lambda (x)
				(if (equal (nth 1 x) suit)
					nil
					t
				)
			)
			cards
		)
	)
)


(defun get-ordered-card-count (cards)
	(let*
		(
			(n-cards
				(list
					(list (count-cards cards 1) 14)
					(list (count-cards cards 2) 2)
					(list (count-cards cards 3) 3)
					(list (count-cards cards 4) 4)
					(list (count-cards cards 5) 5)
					(list (count-cards cards 6) 6)
					(list (count-cards cards 7) 7)
					(list (count-cards cards 8) 8)
					(list (count-cards cards 9) 9)
					(list (count-cards cards 10) 10)
					(list (count-cards cards 11) 11)
					(list (count-cards cards 12) 12)
					(list (count-cards cards 13) 13)
				)
			)
			(ordered
				(sort
					n-cards
					(lambda (x y)
						;sort by number first, value if they're equal
						(if (equal (nth 0 x) (nth 0 y))
							;same count, sort by value
							(if (> (nth 1 x) (nth 1 y))
								t
								nil
							)
							;different count: highest first
							(if (> (nth 0 x) (nth 0 y))
								t
								nil
							)
						)
					)
				)
			)
		)
		ordered
	)
)


(defun get-straight-flush-recursive (n-cards cards cnt)
	(when (equal cnt 11)
		(return-from get-straight-flush-recursive nil)
	)
	;if there's a straight, starting at position (cnt - 1)
	(when (and
					(not (equal (nth (- cnt 1) n-cards) 0))
					(not (equal (nth  cnt  n-cards) 0))
					(not (equal (nth (+ cnt 1) n-cards) 0))
					(not (equal (nth (+ cnt 2) n-cards) 0))	
					(not (equal (nth (+ cnt 3) n-cards) 0))
				)
		;and, we can find equivalent cards of one suit
		(dolist (suit (list `H `D `S `C))
			(when	(and
							(find (list cnt suit) cards :test `equal)
							(find (list (+ cnt 1) suit) cards :test `equal)
							(find (list (+ cnt 2) suit) cards :test `equal)
							(find (list (+ cnt 3) suit) cards :test `equal)
							;check for A at the top
							(if (equal cnt 10)
								(find (list 1 suit) cards  :test `equal)
								(find (list (+ cnt 4) suit) cards  :test `equal)
							)
						)
				;returns highest value in the straight flush
				(return-from get-straight-flush-recursive (+ cnt 4))
			)
		)
	)
	;not at this point, recurse
	(get-straight-flush-recursive n-cards cards (+ cnt 1))
)


(defun get-straight-recursive (n-cards cards cnt)
	(when (equal cnt 11)
		(return-from get-straight-recursive nil)
	)
	;if there's a straight, starting at position (cnt - 1)
	(when (and
					(not (equal (nth (- cnt 1) n-cards) 0))
					(not (equal (nth  cnt  n-cards) 0))
					(not (equal (nth (+ cnt 1) n-cards) 0))
					(not (equal (nth (+ cnt 2) n-cards) 0))	
					(not (equal (nth (+ cnt 3) n-cards) 0))
				)
		;returns highest value in the straight
		(return-from get-straight-recursive (+ cnt 4))
	)
	;not at this point, recurse
	(get-straight-recursive n-cards cards (+ cnt 1))
)

(defun get-straight-flush (n-cards cards)
	;all other hands are easy to determine, done in 
	; "count-cards-and-suit" below
	;straight flush requires additional complexity, so pulled into its own function
	(get-straight-flush-recursive
		 (append n-cards (list (car n-cards)))
		 cards
		 1
	)
)

(defun get-full-house (cards)
	(let
		(
			(n-cards (get-ordered-card-count cards))
		)
		(if (equal (nth 0 (nth 0 n-cards)) 3)
			(if (>= (nth 0 (nth 1 n-cards)) 2)
				t
				nil
			)
			nil
		)
	)
)

(defun get-straight (n-cards cards)
	;all other hands are easy to determine, done in 
	; "count-cards-and-suit" below
	;straight  requires additional complexity, so pulled into its own function
	(get-straight-recursive
		 (append n-cards (list (car n-cards)))
		 cards
		 1
	)
)


(defun get-4-of-a-kind-score (cards)
	(let
		(
			(n-cards (get-ordered-card-count cards))
		)
		(+ 800 
			(* 4 (nth 1 (nth 0 n-cards))) 	
			(nth 1 (nth 1 n-cards))
		)
	)
)
(defun get-full-house-score (cards)
	(let
		(
			(n-cards (get-ordered-card-count cards))
		)
		(+ 700 
			(* 3 (nth 1 (nth 0 n-cards))) 	
			(* 2 (nth 1 (nth 1 n-cards)))
		)
	)
)

(defun get-flush-score (cards)
	(let
		(
			(n-cards (get-ordered-card-count cards))
		)
		(+ 600 
			(nth 1 (nth 0 n-cards)) 	
			(nth 1 (nth 1 n-cards))
			(nth 1 (nth 2 n-cards))
			(nth 1 (nth 3 n-cards))
			(nth 1 (nth 4 n-cards))
		)
	)
)

(defun get-3-of-a-kind-score (cards)
	(let
		(
			(n-cards (get-ordered-card-count cards))
		)
		(+ 400 
			(* 3 (nth 1 (nth 0 n-cards)))
			(if (equal (nth 0 (nth 1 n-cards)) 2) 	
				(* 2 (nth 1 (nth 1 n-cards)))
				(+ 
					(nth 1 (nth 1 n-cards))
					(nth 1 (nth 2 n-cards))
				)
			)
		)
	)
)
(defun get-2-pairs-score (cards)
	(format t "2 pairs: cards ~a~%" cards)
	(let
		(
			(n-cards (get-ordered-card-count cards))
		)
		(format t "Ordered cards ~a~%" n-cards)
		(+ 300 
			(* (/ 90 14) (nth 1 (nth 0 n-cards)))
			(* (/ 9 14) (nth 1 (nth 1 n-cards)))
			(* (/ (/ 9 10) 14) (nth 1 (nth 2 n-cards)))
		)
	)
)
(defun get-1-pair-score (cards)
	(let
		(
			(n-cards (get-ordered-card-count cards))
		)
		(+ 200 
			(* 2 (nth 1 (nth 0 n-cards)))
		  (nth 1 (nth 1 n-cards))
			(nth 1 (nth 2 n-cards))
			(nth 1 (nth 3 n-cards))
		)
	)
)
(defun get-high-card-score (cards)
	(let
		(
			(n-cards (get-ordered-card-count cards))
		)
		(+ 100 
			(nth 1 (nth 0 n-cards))
		  (nth 1 (nth 1 n-cards))
			(nth 1 (nth 2 n-cards))
			(nth 1 (nth 3 n-cards))
			(nth 1 (nth 4 n-cards))
		)
	)
)


(defun count-cards-and-suits (cards)
	(let
		(
			(hearts (count-suit cards `H))
			(diamonds (count-suit cards `D))
			(spades (count-suit cards `S))
			(clubs (count-suit cards `C))
			(n-cards
				(list 
					(count-cards cards 1)
					(count-cards cards 2)
					(count-cards cards 3)
					(count-cards cards 4)
					(count-cards cards 5)
					(count-cards cards 6)
					(count-cards cards 7)
					(count-cards cards 8)
					(count-cards cards 9)
					(count-cards cards 10)
					(count-cards cards 11)
					(count-cards cards 12)
					(count-cards cards 13)
				)
			)
		)
		; Each hand gets assigned a base score of 1-9
		; 	We then multiply by 100, so we get a range of 100 between hands
		;		That allows us to add offset (0-13 per card) based on highest cards left in hand
		;		To distinguish between same hands (e.g., both 2 pairs)		
		(cond
			;straight flush
			((get-straight-flush n-cards cards)
				;we add to score value of highest card in hand
				(+ (* 9 100) (get-straight-flush n-cards cards))
			)
			;four of a kind
			((find 4 n-cards :test `equal)
				 (get-4-of-a-kind-score cards) 
			)
			;full house
			((get-full-house cards)
				;need to add score 3*value of card in trio
				;plus 2*value of card in pair
				(get-full-house-score cards)
			)
			;flush
			((or
				(>= hearts 5)
				(>= diamonds 5)
				(>= clubs 5)
				(>= spades 5)
			 )
				(get-flush-score 
					(remove-if
						(lambda (x)
							(if (not 
										(equal 
											(nth 1 x) 
											(cond
												((>= hearts 5)
													`H
												)
												((>= clubs 5)
													`C
												)
												((>= diamonds 5)
													`D
												)
												((>= spades 5)
													`S
												)
											)
										)
									)
								t
								nil
							)
						)
						cards 
					)
				)
			)
			;straight
			((get-straight n-cards cards)
				(+ (* 5 100) (get-straight n-cards cards))
			)
			;3 of a kind
			((find 3 n-cards :test `equal)
				(get-3-of-a-kind-score cards)
			)
			;2 pairs
			(	(>= (list-length
				(remove nil
					(remove-if
						(lambda (x)
							(if (equal x 2)
								nil
								t
							)
						) 
						n-cards
					)
					:test `equal
				)) 2)				
				(get-2-pairs-score cards)
			)
			;1 pair
			((find 2 n-cards :test `equal)
				(get-1-pair-score cards)
			)
			;nothing: high card
			(t
				(get-high-card-score cards)
			)
		)
	)
)

(defun print-hand (index)
	(let
		(
			(hands 	(list "High-Card" "1 Pair" "2 Pairs" "3 of a Kind"
										"Straight" "Flush" "Full House" "4 of a Kind"
										"Straight Flush"
							)
			)
			(resolved-index 
				(/ (- index (mod index 100)) 100)
			)
		)
		(format t "~a~%"  (nth (- resolved-index 1) hands))
	)
)




;returns true iff:
;	there is a single player in game, or
;	all non-folded players have the same bet or 0 chips left, or
;	all non-folded players have 0 chips available
(defun bets-resolved (game-state)
	;test for single non-folded player
	(if (equal 
				(list-length
					(remove-if 
						(lambda (x)
							(if (equal (nth 3 x) `fold)
								t
								nil
							)
						)
						game-state
					)
				)
				1
			)
		(return-from bets-resolved t)
	)
	;test for bet equality, folded, or 0 chips left
	(let
		(
			(max-bet 
				(nth 2
					(reduce
						(lambda(x y)
							(if (> (nth 2 x) (nth 2 y))
								x
								y
							)
						)
						game-state
					)
				)
			)
		)
		(if (equal 
					(remove-if
						(lambda (x)
							(if (or
										(equal (nth 2 x) max-bet)
										(equal (nth 1 x) 0)
										(equal (nth 3 x) `fold)
									)
								t
								nil
							)
						)
						game-state
					)
					nil
				)
			(return-from bets-resolved t)
		)
	)
	;default nil return value
	nil
)

(defun player-action-update (player cards-on-table game-state)
  (let* ((action (funcall (nth 0 player) cards-on-table game-state))
         (max-bet (nth 2 (reduce (lambda (x y)
                                   (if (> (nth 2 x) (nth 2 y))
                                       x
                                       y))
                                 game-state))))
    (cond
      ;; fold
      ((equal action 'fold)
       (format t "Player ~a folds~%" (nth 0 player))
       (list (nth 0 player)
             (nth 1 player)
             (nth 2 player)
             'fold))

      ;; check / call
      ((equal action 'in)
       (format t "Player ~a calls/checks~%" (nth 0 player))
       (list
        (nth 0 player)
        ;; new chips
        (if (equal (nth 2 player) max-bet)
            (nth 1 player)
            (if (>= (nth 1 player) (- max-bet (nth 2 player)))
                (- (nth 1 player) (- max-bet (nth 2 player)))
                0))
        ;; new bet
        (if (equal (nth 2 player) max-bet)
            (nth 2 player)
            (if (>= (nth 1 player) (- max-bet (nth 2 player)))
                max-bet
                (+ (nth 1 player) (nth 2 player))))
        'call-check))

      ;; integer raise
      ((integerp action)
       (format t "Player ~a raises ~a~%" (nth 0 player) action)
       (list (nth 0 player)
             (- (nth 1 player) action)
             (+ (nth 2 player) action)
             'raise))

      ;; anything else: bad action → force fold
      (t
       (format t "Bad player action ~S from player ~A, forcing fold.~%"
               action (nth 0 player))
       (list (nth 0 player)
             (nth 1 player)
             (nth 2 player)
             'fold)))))

(defun get-players-update-recursive
	(
		players
		max-bet
		cards-on-table
		game-state
	)
	(when (equal players nil)
		(return-from get-players-update-recursive game-state)
	)
	;(format t "Probing players on pot size ~a~%" max-bet)
	(get-players-update-recursive
		(cdr players)
		max-bet
		cards-on-table
		(mapcar
			(lambda (player)
				(if (equal (nth 0 player) (car players))
					(cond
						((equal (nth 3 player) `fold)
							player
						)
						((equal (nth 3 player) `raise)
							;has anyone else raised? update
							;no, change state to in
							(if (not (equal (nth 2 player) max-bet))
								(player-action-update player cards-on-table game-state)
								(list
									(nth 0 player)
									(nth 1 player)
									(nth 2 player)
									`in
								)
							)
						)
						((equal (nth 3 player) `call-check)
							;has anyone raised? update
							;no, change state to in
							(if (not (equal (nth 2 player) max-bet))
								(player-action-update player cards-on-table game-state)
								(list
									(nth 0 player)
									(nth 1 player)
									(nth 2 player)
									`in
								)
							)
						)
						(t ;default for players already in: what to do?
							(player-action-update player cards-on-table game-state)
						)
					)
					;else: player
					player
				)
			)
			game-state
		)
	)
)

(defun get-players-update (cards-on-table game-state)
	(let*
		(
			(max-bet 
				(nth 2 
					(reduce 
						(lambda(x y) 
							(if (> (nth 2 x) (nth 2 y)) x y )
					 	) 
						game-state 
					) 
				) 
			)
			(players-list
				(mapcar
					(lambda (x)
						(nth 0 x)
					)
					game-state
				) 
			)
		)
		;(format t "Getting players' moves...~%")
		(get-players-update-recursive 
			players-list 
			max-bet
			cards-on-table 
			game-state
		)
	)
)


(defun resolve-betting (cards-on-table game-state)
		(when (bets-resolved game-state)
			;(format t "Bets resolved, state ~%" game-state)
			;all bets same or players folded, we're good to go
			(return-from resolve-betting game-state)
		)
		;(format t "Bets not resolved, going through players...~%")
		;go through one round of players' updates, then recurse
		(resolve-betting 
			cards-on-table
			(get-players-update 
				cards-on-table 
				game-state
			)
		)
)

(defun game-non-blind (cards-on-table game-state)
	;we're called after dealing cards, players have a chance to bet
	;after that, resolve betting same as after blinds
	(resolve-betting 
		cards-on-table
		(get-players-update cards-on-table 
			(mapcar	
				(lambda (x)
					(list
						(nth 0 x)
						(nth 1 x)
						(nth 2 x)
						(if (equal (nth 3 x) `fold)
							`fold
							`in
						)
					)
				)
				game-state
			)		
		)
	)
)

(defun game-blind (cards-on-table game-state)
	;easiest way, assume last two elements are blinds
	;last element is big blind, penultimate is big blind
	;that way, we always start resolving bets from first in list
	;same whether it's 2 or more players
	(let*
		(
			;so last two elements are first two
			(reversed (reverse game-state))
			(big-blind (nth 0 reversed))
			(small-blind (nth 1 reversed))
			(remainder (cdr (cdr reversed)))
			;add blinds to first two
			(reversed-with-blind
				(append
					(list
						;big blind
						(list
							(nth 0 big-blind)
							(if (>= (nth 1 big-blind) 40)
								(- (nth 1 big-blind) 40)
								0
							)
							(if (>= (nth 1 big-blind) 40)
								40
								(nth 1 big-blind)
							)
							`in
						)
						;small blind
						(list
							(nth 0 small-blind)
							(if (>= (nth 1 small-blind) 20)
								(- (nth 1 small-blind) 20)
								0
							)
							(if (>= (nth 1 small-blind) 20)
								20
								(nth 1 small-blind)
							)
							`in
						)
					)
					remainder
				)
			)	
		)
		;call resolve-bets, after reversing list with blinds again
		(resolve-betting
			cards-on-table
			(mapcar	
				(lambda (x)
					(list
						(nth 0 x)
						(nth 1 x)
						(nth 2 x)
						(if (equal (nth 3 x) `fold)
							`fold
							`in
						)
					)
				)
				(reverse reversed-with-blind)
			)		
		)
	)
)

(declaim (ftype function payout))
(declaim (ftype function pay-all-to-one))

;returns total pot for this round so far
(defun get-pot (game-state)
	(let
		(
			(pot 0)
		)
		(dolist (player game-state)
			(setf pot
				(+ pot (nth 2 player))
			)
		)
		pot
	)
)



(defun resolve-pay (players-scores game-state cards-on-table players-hands-get)
	(when (equal (get-pot game-state) 0)
		;we've finished distributing payouts
		(return-from resolve-pay game-state)
	)
	(let*
		(
			;find players that need to be paid first
			(high-score 
				(reduce 
					(lambda (x y)
						(if (> x y)
							x
							y
						)
					)
					players-scores
				)
			)
			(high-score-players
				(mapcar
					(lambda (x)
						(if (equal x high-score)
							x
							nil
						)
					)
					players-scores
				)
			)
			;how many players to pay?
			(n-to-pay
				(list-length
					(remove-if
						(lambda (x)	
							(if (not (equal x high-score))
								t
								nil
							)		
						)
						players-scores	
					)
				)		
			)
			;total amount to be deducted per bet and awarded to high-score players
			;smallest ammount of high score players
			(deduction
				(reduce
					(lambda (x y)
						(cond
							((> x y)
								(if (equal y 0)
									x
									y
								)
							)
							((> y x)
								(if (equal x 0)
									y
									x
								)
							)	
							(t x
							)
						)
					)
					(mapcar
						(lambda (x y)
							(if y
								(nth 2 x)
								0
							)
						)
						game-state
						high-score-players
					)
				)
			)
			(pot-to-split
				(reduce
					`+
					(mapcar
						(lambda (x)
							(if (> (nth 2 x) deduction)
								deduction
								(nth 2 x)
							)
						)
						game-state
					)
				)
			)
		)
		;(format t "Have high score players ~a~%" high-score-players)
		(let
			(
				(updated-state
					(mapcar	
						(lambda (player score)
							(if score
								;high score player: we're sure bet is at least deduction
								(list
									(nth 0 player)
									(+ (nth 1 player) (/ pot-to-split n-to-pay))
									(- (nth 2 player) deduction)
									(nth 3 player)
								)	
								;not high score player, just pays (may be less than deduction)
								(list
									(nth 0 player)
									(nth 1 player)
									(if (> (nth 2 player) deduction)
										(- (nth 2 player) deduction)
										0
									)
									(nth 3 player)
								)
							)
						)
						game-state
						high-score-players
					)
				)	
			)
			;recursive call with updated lists 
			;call "payout" again
			(payout updated-state cards-on-table players-hands-get)
		)
	)
)

(defun pay-all-to-one (game-state)
	(let
		(
			(pot (get-pot game-state))
		)
		(mapcar
			(lambda (x)
				(list
					(nth 0 x)
					(if (equal (nth 3 x) `fold)
						(nth 1 x)
						(+ (nth 1 x) pot)
					)
					0
					(nth 3 x)
				)
			)
			game-state
		)
	)
)




;called at the end of a round to collect pot and award to winner(s)
(defun payout (game-state cards-on-table players-hands-get)
	;for debug purposes:
	(when (find-if (lambda(x) (if (< (nth 1 x) 0) t nil)) game-state)
		(format t "Error: ~a~%" game-state)
		(read)
	)
	;deal with special case where everyone else folded
	(when (equal 
					(list-length 
						(remove-if 
							(lambda(x) (if (equal (nth 3 x) `fold) t nil))
							 game-state
						)
					) 					
					1
				)
		(return-from payout (pay-all-to-one game-state))
	)
	(let*
		(
			(players-scores 
				(mapcar 
					(lambda (hand-fn state)
						(if (or 
									(equal (nth 3 state) `fold)
									(equal (nth 2 state) 0)
								)
							0
							;else, in play
							(progn
								;(format t "State ~a~%" game-state)
								(format t "Player ~a has hand: " (nth 0 state))
								(print-cards (funcall hand-fn))
								(format t "->") 
								(print-hand
									(count-cards-and-suits  
										(append cards-on-table (funcall hand-fn))	
									)
								)
								(format t "~%")
								(count-cards-and-suits  
									(append cards-on-table (funcall hand-fn))	
								)
							)
						)
					)
					players-hands-get
					game-state
				)
			)
		)
		(when (equal
						1
						(list-length 
							(remove-if
								(lambda (x)
									(if (equal (nth 3 x) `fold)
										t
										nil
									) 
								)
								game-state
							)
						) 
					)
		)
		(resolve-pay 
			players-scores 
			game-state 
			cards-on-table 
			players-hands-get
		)
	)
)



;Single round  game engine
;receives as arguments:
;		"game-state"
;		is a list of size #(number of players)
;		each item in the list is also a list, of format:
;			(`player-function chips current-bet state)
;		i.e., the function to call to determine player action
;   total chips the player has, and current bet
;		and current playing state ("in" or "fold")   
; for example, for 3 players, it might look like:
;	(list
;		(list `player1 900  50 `in) ;has 900 chips, 50 bet, playing
;		(list `player2 1000 50 `in) ;has 1000 chips, 50 bet, playing
;		(list `player3 200  40 `fold) ;has 200 chips, 40 bet, folded
;	)
;
(defun game-round (game-state players-hands-deal players-hands-get)
	(when (< (list-length game-state) 2)
		(format t "Error: need at least 2 players.~%")
		(return-from game-round nil)
	)
	(let
		(
			(cards-on-table nil)
		) 
		;initialize deck
		(init-deck)
		(format t "Shuffled deck~%")
		;deal hands to players
		(dolist (player players-hands-deal)
			(funcall player (get-hand))
		)
		(format t "Dealt hands to players~%")
		;collect blinds and initial bets
		(setf game-state (game-blind nil game-state))
		(format t "Bets finished, pot is ~a~%~%" (get-pot game-state))
		;draw flop
		(setf cards-on-table (get-flop))
		(format t "Flop is ")
		(print-cards cards-on-table)
		;collect post-flop bets
		(setf game-state (game-non-blind cards-on-table game-state))
		(format t "Bets finished, pot is ~a~%~%" (get-pot game-state))
		;draw turn
		(setf cards-on-table 
			(append 
				cards-on-table
				(list (get-turn-or-river))
			)
		)
		(format t "After turn, have ")
		(print-cards cards-on-table)
		;collect post-turn bets
		(setf game-state (game-non-blind cards-on-table game-state))
		(format t "Bets finished, pot is ~a~%~%" (get-pot game-state))
		;draw river
		(setf cards-on-table
			(append 
				cards-on-table
				(list (get-turn-or-river))
			)
		)
		(format t "After river, have " )
		(print-cards cards-on-table)
		;collect final bets
		(setf game-state (game-non-blind cards-on-table game-state))
		(format t "Bets finished, pot is ~a~%~%" (get-pot game-state))
		;decide winner, give payout, return updated state
		(payout game-state cards-on-table players-hands-get)
	)
)

(defun full-game (game-state players-hands-deal players-hands-get)
	;game is over when only one player has non-zero chips
	(when (equal 
					(list-length
						(remove-if
							(lambda (x)
								(if (equal (nth 1 x) 0)
									t
									nil
								)
							)
							game-state
						)
					)
					1
				)
		(format t "Game finished: ~a takes all~%"
			(nth 0
				(find-if 
					(lambda (x) 
						(if (not (equal (nth 1 x) 0))
							t
							nil	
						)
					)
					game-state
				)
			)
		)	
		(return-from full-game nil)
	)
	(let
		(
			;updated state after one round
			(updated-state 
				;remove any players that now have 0 chips
				(remove-if
					(lambda (x)
						(if (equal (nth 1 x) 0)
							t
							nil
						)
					)
					(game-round game-state players-hands-deal players-hands-get)
				)
			)
		)
		(format t "~a~%" updated-state)
		(full-game 
			;must shift first element to the end
			(append (cdr updated-state) (list (car updated-state)))
			(append (cdr players-hands-deal) (list (car players-hands-deal)))
			(append (cdr players-hands-get) (list (car players-hands-get)))
		)
	)
)





