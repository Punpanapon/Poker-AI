Files:
	"poker.lisp": main game engine. You do not need to modify nor read this file, unless you're looking for more details on the game state format.
	
	"players.lisp": this is where your logic must go: there is a skeleton of a function named "AI" which is the main entry point for your logic.

	"main.lisp": main program entry point. Provides three functions to test a single round, using the default players; a full game, using the default players; and, a full game using the default players + your AI.

Usage:

	If using sbcl, "sbcl --load main.lisp" will load the entire program into a running Lisp image. Functions in "main" can then be called to test behavior (e.g., "(test-1-round)").
	
	Note 1: sbcl will probably give you 2 style warnings, saying variables in "AI" are not used. These will disappear once you use them.	

	Note 2: program is printing Unicode characters for card suit display. If your terminal does not support Unicode, either upgrade to a terminal that does (it is 2025...) or change the function in "poker.lisp" that prints the suit to just display a letter.

Suggestions:

	Using only the content covered in this course, you may build a poker-playing AI by:

		1-Probabilistic MiniMax. Easiest option, but won't play very well and will probably stress your computing power, given any non-trivial depth. Deals with probable outcomes, but cannot account for other players bluffing.

	2-Hidden Markov Models on a Bayesian Network. Can model other players' behaviors (e.g. bluffing) as a hidden variable, and MCMC can be used to determine chances in a more efficient way than option (1).

	3-Markov Decision Process. Can compute situations offline and deploy very efficient probabilistic computations, but does not account for bluffing.

	4-Combinations of all the above (this is the best case solution, where you use your ingenuity to combine different technologies in hopefully surprising ways, yielding good results).

	5-Using other methods that are not part of the syllabus (e.g., Deep Learning). Still must be done in Common Lisp, though...

