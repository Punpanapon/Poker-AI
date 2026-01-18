;;; players.lisp - AI and sample players for the poker engine

;;;; ------------------------------------------------------------------
;;;; Global state for AI player
;;;; ------------------------------------------------------------------

(defparameter *ai-hand* nil)          ; our 2-card hand
(defparameter *decision-log* nil)     ; list of decision log entries

;; HMM-style opponent modeling
(defparameter *opponent-states* '(:tight :loose))
(defparameter *opponent-beliefs* (make-hash-table)) ; name -> plist (:tight p :loose q)
(defparameter *opponent-emissions*
  '((:tight . (:fold 0.5 :call 0.4 :raise-small 0.09 :raise-big 0.01))
    (:loose . (:fold 0.1 :call 0.4 :raise-small 0.3  :raise-big 0.2))))

;; Long-term opponent statistics (across many hands)
(defparameter *opponent-stats* (make-hash-table))
;; name -> plist (:observations n :fold n :call n :raise-small n :raise-big n)

(defun get-opponent-stats (name)
  (or (gethash name *opponent-stats*)
      (setf (gethash name *opponent-stats*)
            (list :observations 0
                  :fold 0 :call 0
                  :raise-small 0 :raise-big 0))))

(defun update-opponent-stats (name obs)
  (let ((st (get-opponent-stats name)))
    (incf (getf st :observations))
    (ecase obs
      (:fold        (incf (getf st :fold)))
      (:call        (incf (getf st :call)))
      (:raise-small (incf (getf st :raise-small)))
      (:raise-big   (incf (getf st :raise-big))))
    (setf (gethash name *opponent-stats*) st)))

(defun opponent-style-from-stats (stats)
  (let* ((n (max 1 (getf stats :observations)))
         (folds (getf stats :fold))
         ;; VPIP ~ fraction of non-fold actions
         (vpip (/ (float (- n folds)) n))
         (raises (/ (float (+ (getf stats :raise-small)
                              (getf stats :raise-big)))
                    n)))
    (cond
      ;; very tight, almost never enters pots
      ((and (< vpip 0.15) (< raises 0.08)) :nit)
      ((< vpip 0.25)                        :tight)
      ;; standard loose fish / maniac
      ((> vpip 0.55)                        :very-loose)
      ((> vpip 0.40)                        :loose)
      (t                                    :normal))))

(defun main-opponent-style (game-state)
  (let* ((active-opponents
          (remove-if (lambda (pl)
                       (or (equal (first pl) 'AI)
                           (equal (fourth pl) 'FOLD)))
                     game-state))
         (target (first active-opponents)))
    (if (null target)
        :normal
        (opponent-style-from-stats
         (get-opponent-stats (first target))))))



;;;; ------------------------------------------------------------------
;;;; MDP for abstract strategy (offline policy)
;;;; ------------------------------------------------------------------

(defparameter *mdp-abstract-states* nil)
(defparameter *mdp-utilities* (make-hash-table :test 'equal))
(defparameter *mdp-policy* (make-hash-table :test 'equal))
(defparameter *mdp-gamma* 0.8)    ; discount factor
(defparameter *mdp-initialized* nil)
(defparameter *mdp-actions* '(:fold :call :raise))

;;;; ------------------------------------------------------------------
;;;; Interface functions expected by the engine for the AI player
;;;; ------------------------------------------------------------------

(defun AI-set-hand (cards)
  "Store our 2 hole cards for this hand."
  (setf *ai-hand* cards))

(defun AI-get-hand ()
  "Return our stored hole cards."
  *ai-hand*)

;;;; ------------------------------------------------------------------
;;;; Hand strength heuristic (very simple, 2-card only)
;;;; ------------------------------------------------------------------

(defun hand-strength ()
  "Return a simple numeric rating of our 2-card starting hand.
Larger numbers mean stronger hands. This is a heuristic only."
  (let* ((c1 (first *ai-hand*))
         (c2 (second *ai-hand*))
         (r1 (first c1))
         (r2 (first c2))
         (s1 (second c1))
         (s2 (second c2))
         ;; card ranks are 1–13 (A..K); treat Ace (1) as 14
         (v1 (if (= r1 1) 14 r1))
         (v2 (if (= r2 1) 14 r2))
         (pairp (= r1 r2))
         (suitedp (equal s1 s2))
         (high-rank (max v1 v2))
         (gap (abs (- v1 v2))))
    (+ (if pairp 40 0)                 ; big bonus for a pair
       (cond                           ; high card bonus
         ((>= high-rank 13) 25)        ; K or A
         ((>= high-rank 11) 15)        ; J or Q
         (t 0))
       (if suitedp 10 0)               ; suited bonus
       (cond                           ; straight-ish bonus
         ((<= gap 1) 10)               ; connectors / 1-gap
         ((<= gap 2) 5)
         (t 0))
       (- high-rank 8))))              ; small linear factor

;;;; ------------------------------------------------------------------
;;;; Decision-state representation
;;;; ------------------------------------------------------------------

(defstruct decision-state
  street              ; :preflop, :flop, :turn, :river
  pot-size            ; approximate current pot size
  num-players-active  ; how many players have not folded
  my-position         ; our index among active players (0-based)
  my-stack            ; our chip count
  effective-stack     ; min(our stack, largest opponent stack in hand)
  to-call             ; chips needed to call current max bet
  last-raise          ; last raise size (0 if unknown)
  hole-cards          ; our two cards
  board-cards         ; community cards
  hand-strength       ; numeric heuristic from (hand-strength)
  pot-odds)           ; to-call / pot-size (0 if pot-size <= 0)

(defun decision-state-cards-on-table (state)
  (decision-state-board-cards state))

;; keep a last-known AI stack so a transient missing row doesn't make us act broke
(defparameter *last-known-ai-stack* 1000)

(defun extract-decision-state (cards-on-table game-state)
  (let* (;; determine street by number of board cards
         (street (cond ((= (length cards-on-table) 0) :preflop)
                       ((= (length cards-on-table) 3) :flop)
                       ((= (length cards-on-table) 4) :turn)
                       ((= (length cards-on-table) 5) :river)
                       (t :unknown)))

         ;; find our own state in game-state: (AI chips bet status)
         (my-state (find-if (lambda (x) (equal (first x) 'AI)) game-state))

         ;; stack: use last-known fallback if AI row missing or bad
         (my-stack (if (and my-state (numberp (second my-state)))
                       (progn
                         (setf *last-known-ai-stack* (second my-state))
                         (second my-state))
                       *last-known-ai-stack*))

         ;; current bet
         (my-bet (if (and my-state (numberp (third my-state)))
                     (third my-state)
                     0))

         ;; max bet at the table (uses safe-bet so NIL rows are treated as 0)
         (max-bet (reduce #'max
                          (mapcar #'safe-bet game-state)
                          :initial-value 0))
         (raw-to-call (- max-bet my-bet))
         (to-call (if (> raw-to-call 0) raw-to-call 0))

         ;; active players (status not FOLD), skip NIL rows
         (active-players
           (remove-if (lambda (x)
                        (or (null x)
                            (equal (fourth x) 'FOLD)))
                      game-state))
         (num-active (length active-players))

         ;; our position among active players
         (my-position (or (position 'AI active-players :key #'first) 0))

         ;; opponent stacks: ignore AI and any non-numeric stacks
         (raw-opp-rows
           (remove-if (lambda (x)
                        (or (null x)
                            (equal (first x) 'AI)))
                      active-players))
         (opp-stacks
           (remove-if-not #'numberp
                          (mapcar (lambda (row) (and row (second row)))
                                  raw-opp-rows)))
         (max-opp-stack (if opp-stacks
                            (reduce #'max opp-stacks)
                            my-stack))
         (effective-stack (min my-stack max-opp-stack))

         ;; pot size = sum of all current bets
         (pot-size (let ((bets (mapcar #'safe-bet game-state)))
                     (reduce #'+ bets :initial-value 0)))

         ;; pot odds: to-call / pot-size (or 0 if pot-size <= 0)
         (pot-odds (if (> pot-size 0)
                       (/ (float to-call) pot-size)
                       0.0))

         ;; TODO: track last raise size properly; placeholder 0 for now
         (last-raise 0)

         ;; hand-strength heuristic
         (hs (hand-strength)))
    (make-decision-state
     :street street
     :pot-size pot-size
     :num-players-active num-active
     :my-position my-position
     :my-stack my-stack
     :effective-stack effective-stack
     :to-call to-call
     :last-raise last-raise
     :hole-cards *ai-hand*
     :board-cards cards-on-table
     :hand-strength hs
     :pot-odds pot-odds)))


(defun safe-bet (player-state)
  (let ((b (third player-state)))
    (if (numberp b) b 0)))

;;;; ------------------------------------------------------------------
;;;; Decision logging
;;;; ------------------------------------------------------------------

(defun log-decision (bot-name state action)
  (let ((entry (list
                :time          (get-universal-time)
                :bot           bot-name
                :street        (decision-state-street state)
                :pot           (decision-state-pot-size state)
                :my-stack      (decision-state-my-stack state)
                :to-call       (decision-state-to-call state)
                :hand-strength (decision-state-hand-strength state)
                :action        action)))
    (push entry *decision-log*)
    *decision-log*))

(defun clear-decision-log ()
  (setf *decision-log* nil))

;;;; ------------------------------------------------------------------
;;;; HMM opponent modeling helpers
;;;; ------------------------------------------------------------------

(defun get-opponent-belief (name)
  (or (gethash name *opponent-beliefs*)
      (setf (gethash name *opponent-beliefs*)
            (list :tight 0.5 :loose 0.5))))

(defun emission-prob (state observation)
  (let* ((pair (assoc state *opponent-emissions*))
         (plist (cdr pair)))
    (or (and plist (getf plist observation)) 0.01))) ; small floor

(defun update-opponent-belief (name observation)
  (let* ((prior   (get-opponent-belief name))
         (p-tight (getf prior :tight))
         (p-loose (getf prior :loose))
         (like-tight (emission-prob :tight observation))
         (like-loose (emission-prob :loose observation))
         (unnorm-tight (* like-tight p-tight))
         (unnorm-loose (* like-loose p-loose))
         (z (+ unnorm-tight unnorm-loose)))
    (setf (gethash name *opponent-beliefs*)
          (if (<= z 0)
              (list :tight 0.5 :loose 0.5)
              (list :tight (/ unnorm-tight z)
                    :loose (/ unnorm-loose z))))))

(defun classify-opponent-observation (player-state pot-size max-bet)
  (let* ((status (and player-state (fourth player-state)))
         (bet    (safe-bet player-state)))
    (cond
      ((equal status 'FOLD) :fold)
      ((= max-bet 0) :call)
      (t
       (let* ((ratio (if (> pot-size 0)
                         (/ (float bet) pot-size)
                         0.0)))
         (cond
           ((or (= bet 0) (< ratio 0.2)) :call)
           ((< ratio 0.6)               :raise-small)
           (t                           :raise-big)))))))


(defun update-all-opponent-beliefs (game-state)
  (let* ((pot-size (reduce #'+ game-state :key #'safe-bet :initial-value 0))
         (max-bet  (reduce #'max game-state :key #'safe-bet :initial-value 0)))
    (dolist (pl game-state)
      (let ((name (first pl)))
        (unless (equal name 'AI)
          (let ((obs (classify-opponent-observation pl pot-size max-bet)))
            (update-opponent-belief name obs)
            (update-opponent-stats  name obs)))))
    *opponent-beliefs*))

(defun main-opponent-belief (game-state)
  (let* ((active-opponents
          (remove-if (lambda (pl)
                       (or (equal (first pl) 'AI)
                           (equal (fourth pl) 'FOLD)))
                     game-state)))
    (if (null active-opponents)
        ;; No active opponents: default neutral belief
        (list :tight 0.5 :loose 0.5 :style :balanced :name nil)
        (let* (;; choose the opponent with the largest current bet as the "main" villain
               (target (car (sort (copy-list active-opponents)
                                  #'>
                                  :key #'safe-bet)))
               (name   (first target))
               (hmm    (get-opponent-belief name))
               (stats  (get-opponent-stats   name))
               (style  (opponent-style-from-stats stats))
               (p-tight (getf hmm :tight))
               (p-loose (getf hmm :loose)))
          ;; Slightly tweak HMM probs based on long-term style
          (cond
            ((eq style :nit)
             (setf p-tight (min 0.95 (+ p-tight 0.15))
                   p-loose (max 0.05 (- p-loose 0.15))))
            ((eq style :station)
             (setf p-loose (min 0.95 (+ p-loose 0.15))
                   p-tight (max 0.05 (- p-tight 0.15))))
            ((eq style :aggro)
             (setf p-loose (min 0.95 (+ p-loose 0.10))
                   p-tight (max 0.05 (- p-tight 0.10)))))
          ;; renormalize to sum to 1
          (let* ((z (+ p-tight p-loose))
                 (pt (if (> z 0) (/ p-tight z) 0.5))
                 (pl (if (> z 0) (/ p-loose z) 0.5)))
            (list :tight pt :loose pl :style style :name name))))))

;;;; ------------------------------------------------------------------
;;;; MDP state abstraction helpers
;;;; ------------------------------------------------------------------

(defun hs-bucket (hs)
  (cond
    ((>= hs 70) :monster)
    ((>= hs 50) :strong)
    ((>= hs 30) :medium)
    (t          :weak)))

(defun pot-bucket (pot stack)
  (let ((ratio (if (> stack 0)
                   (/ (float pot) stack)
                   0.0)))
    (cond
      ((< ratio 0.25) :small)
      ((< ratio 0.75) :medium)
      (t              :big))))

(defun opp-type-bucket-from-belief (belief)
  (let ((p-tight (getf belief :tight))
        (p-loose (getf belief :loose)))
    (cond
      ((> p-tight 0.7) :tight)
      ((> p-loose 0.7) :loose)
      (t                :neutral))))

(defun abstract-state-from (state belief)
  (let* ((street (decision-state-street state))
         (hs     (decision-state-hand-strength state))
         (pot    (decision-state-pot-size state))
         (stack  (decision-state-my-stack state))
         (hs-b   (hs-bucket hs))
         (pot-b  (pot-bucket pot stack))
         (opp-b  (opp-type-bucket-from-belief belief)))
    (list :street street
          :hs-bucket hs-b
          :pot-bucket pot-b
          :opp-type opp-b)))

;;;; ------------------------------------------------------------------
;;;; MDP state space and reward function
;;;; ------------------------------------------------------------------

(defun build-mdp-abstract-states ()
  (setf *mdp-abstract-states*
        (loop for street in '(:preflop :flop :turn :river)
              nconc (loop for hs in '(:weak :medium :strong :monster)
                          nconc (loop for pot in '(:small :medium :big)
                                      nconc (loop for opp in '(:tight :neutral :loose)
                                                  collect (list :street street
                                                                :hs-bucket hs
                                                                :pot-bucket pot
                                                                :opp-type opp)))))))

(defun mdp-reward (astate action)
  (let* ((hs   (getf astate :hs-bucket))
         (pot  (getf astate :pot-bucket))
         (opp  (getf astate :opp-type))
         ;; base value from hand strength
         (base (ecase hs
                 (:monster 5.0)
                 (:strong  2.0)
                 (:medium  0.0)
                 (:weak   -2.0)))
         ;; pot effect: big pots amplify outcomes
         (pot-val (ecase pot
                    (:small  0.5)
                    (:medium 1.0)
                    (:big    2.0)))
         ;; opponent effect: against loose villains, value of good hands goes up
         (opp-val (ecase opp
                    (:tight   -0.5)
                    (:neutral  0.0)
                    (:loose    0.5))))
    (case action
      (:fold
       ;; folding sacrifices equity but avoids risk; worse in strong/big-pot spots
       (- (* 0.5 base) (* 0.5 pot-val)))
      (:call
       ;; calling takes medium risk, medium reward
       (+ (* 1.0 base) (* 0.5 pot-val) opp-val))
      (:raise
       ;; raising invests more, but can win more; penalize a little for risk
       (+ (* 1.5 base) (* 1.0 pot-val) opp-val -1.0))
      (otherwise 0.0))))

;;;; ------------------------------------------------------------------
;;;; MDP value iteration and policy extraction
;;;; ------------------------------------------------------------------

(defun initialize-mdp ()
  (build-mdp-abstract-states)
  ;; initialize U(s) = 0 for all states
  (dolist (s *mdp-abstract-states*)
    (setf (gethash s *mdp-utilities*) 0.0))
  ;; value iteration
  (dotimes (iter 25)  ; 25 iterations is enough for this small model
    (dolist (s *mdp-abstract-states*)
      (let* ((u-old (gethash s *mdp-utilities*))
             (r-fold  (mdp-reward s :fold))
             (r-call  (mdp-reward s :call))
             (r-raise (mdp-reward s :raise))
             (q-fold  r-fold)
             (q-call  (+ r-call (* *mdp-gamma* u-old)))
             (q-raise (+ r-raise (* *mdp-gamma* u-old)))
             (u-new   (max q-fold q-call q-raise)))
        (setf (gethash s *mdp-utilities*) u-new))))
  ;; extract policy π(s) = argmax_a Q(s,a)
  (dolist (s *mdp-abstract-states*)
    (let* ((u (gethash s *mdp-utilities*))
           (r-fold  (mdp-reward s :fold))
           (r-call  (mdp-reward s :call))
           (r-raise (mdp-reward s :raise))
           (q-fold  r-fold)
           (q-call  (+ r-call (* *mdp-gamma* u)))
           (q-raise (+ r-raise (* *mdp-gamma* u)))
           (best-action :fold)
           (best-value q-fold))
      (when (> q-call best-value)
        (setf best-value q-call
              best-action :call))
      (when (> q-raise best-value)
        (setf best-value q-raise
              best-action :raise))
      (setf (gethash s *mdp-policy*) best-action)))
  (setf *mdp-initialized* t)
  *mdp-policy*)

(defun ensure-mdp-initialized ()
  "Run MDP initialization once."
  (unless *mdp-initialized*
    (initialize-mdp)))

(defun mdp-recommend-action (state belief)
  (ensure-mdp-initialized)
  (let* ((astate (abstract-state-from state belief))
         (act    (gethash astate *mdp-policy*)))
    (or act :call)))

;;;; ------------------------------------------------------------------
;;;; Monte Carlo equity estimation
;;;; ------------------------------------------------------------------

(defun safe-score (x)
  (if (numberp x) x 0.0))

(defparameter *strong-mc-samples* 500)

(defun all-cards ()
  (loop for suit in '(H D C S)
        append (loop for v from 1 to 13
                     collect (list v suit))))

(defun remaining-deck (board my-hand)
  (remove-if
   (lambda (card)
     (or (member card board :test #'equal)
         (member card my-hand :test #'equal)))
   (all-cards)))

;; ---- safer draw-random-cards ----
(defun draw-random-cards (deck n)
  (let ((d (copy-list (or deck '())))
        (result '()))
    (when (and (numberp n) (< n 0))
      (error n))
    (dotimes (i (min (length d) (or n 0)))
      (let* ((len (length d))
             ;; ensure random arg >= 1: if len==0 we won't reach here because dotimes count is 0
             (idx (if (> len 0) (random len) 0))
             (card (nth idx d)))
        (setf d (remove card d :count 1 :test #'equal))
        (push card result)))
    (values (nreverse result) d)))

;; ---- safer mc-simulate-one (guards deck exhaustion and uses safe-score) ----
(defun mc-simulate-one (board my-hand)
  (let* ((known-board (or board '()))
         (known-count (length known-board))
         (deck (remaining-deck known-board my-hand))
         (need (max 0 (- 5 known-count))))
    (multiple-value-bind (extra-board deck-after-board)
        (if (> need 0)
            (draw-random-cards deck need)
            (values '() deck))
      ;; If the deck is unexpectedly too small to continue, treat as tie-safe outcome.
      (unless deck-after-board
        (return-from mc-simulate-one :tie))
      (let* ((final-board (append known-board extra-board))
             ;; draw up to 2 cards for the opponent; handle if deck small
             (opp-hand (multiple-value-bind (cards new-deck)
                           (draw-random-cards deck-after-board 2)
                         (declare (ignore new-deck))
                         cards))
             (my-score  (safe-score (and (fboundp 'count-cards-and-suits)
                                        (count-cards-and-suits (append final-board my-hand)))))
             (opp-score (safe-score (and (fboundp 'count-cards-and-suits)
                                         (count-cards-and-suits (append final-board opp-hand))))))
        (cond
          ((> my-score opp-score) :win)
          ((< my-score opp-score) :loss)
          (t                      :tie))))))

(defun estimate-win-probability-mc (board my-hand &key (samples *strong-mc-samples*))
  (let ((wins 0)
        (losses 0)
        (ties 0))
    (dotimes (i samples)
      (case (mc-simulate-one board my-hand)
        (:win (incf wins))
        (:loss (incf losses))
        (:tie (incf ties))))
    (let ((total (+ wins losses ties)))
      (if (zerop total)
          0.5
          (/ (+ wins (* 0.5 ties))
             total)))))

(defun mc-simulate-one-multi (board my-hand num-opps)
  (let* ((known-board board)
         (known-count (length known-board))
         (deck (remaining-deck known-board my-hand))
         (need (- 5 known-count)))
    (multiple-value-bind (extra-board deck-after-board)
        (if (> need 0)
            (draw-random-cards deck need)
            (values '() deck))
      (let* ((final-board (append known-board extra-board))
             (my-score (safe-score (count-cards-and-suits
                                    (append final-board my-hand))))
             (best-opp-score -1.0)
             (current-deck deck-after-board))
        ;; Draw a separate random hand for each opponent
        (dotimes (i num-opps)
          (multiple-value-bind (opp-hand new-deck)
              (draw-random-cards current-deck 2)
            (setf current-deck new-deck)
            (let ((opp-score (safe-score (count-cards-and-suits
                                          (append final-board opp-hand)))))
              (when (> opp-score best-opp-score)
                (setf best-opp-score opp-score)))))
        (cond
          ((> my-score best-opp-score) :win)
          ((< my-score best-opp-score) :loss)
          (t                           :tie))))))

(defun estimate-win-probability-mc-multi (board my-hand num-opps &key (samples 200))
  (let ((wins 0)
        (losses 0)
        (ties 0))
    (dotimes (i samples)
      (case (mc-simulate-one-multi board my-hand num-opps)
        (:win (incf wins))
        (:loss (incf losses))
        (:tie (incf ties))))
    (let ((total (+ wins losses ties)))
      (if (zerop total)
          0.5
          (/ (+ wins (* 0.5 ties))
             total)))))

;;;; ------------------------------------------------------------------
;;;; Dynamic bet sizing helpers
;;;; ------------------------------------------------------------------

(defun choose-preflop-raise-amount (win-prob stack)
  (let* ((stack (if (and stack (numberp stack)) stack 0))
         (base
           (cond
             ;; Huge preflop edge (AA/KK/QQ-type equity vs 3 players)
             ((>= win-prob 0.75) 90)
             ;; Very strong
             ((>= win-prob 0.65) 70)
             ;; Strong
             ((>= win-prob 0.55) 50)
             ;; Decent
             ((>= win-prob 0.45) 40)
             ;; Marginal, if we raise at all
             (t                   30)))
         ;; we *want* to keep some chips, but never ask for more than we have
         (max-afford (min stack (max 20 (- stack 200))))
         (amount (min base max-afford)))
    ;; final clamp: never more than stack, never negative
    (max 0 (min amount stack))))

(defun choose-postflop-raise-amount (win-prob stack pot-size)
  (let* ((stack (if (and stack (numberp stack)) stack 0))
         (pot-size (if (and pot-size (numberp pot-size)) pot-size 0))
         (base
           (cond
             ;; Monster hands / huge equity: build a big pot
             ((>= win-prob 0.85)
              (+ 60 (truncate (/ pot-size 3))))
             ;; Strong hands: healthy value bet
             ((>= win-prob 0.70)
              (+ 40 (truncate (/ pot-size 4))))
             ;; Medium ~ good equity: normal raise
             ((>= win-prob 0.55)
              40)
             ;; Bluffs / thin spots: keep it smaller
             (t
              30)))
         ;; never more than we actually have
         (max-afford (min stack (max 20 (- stack 150))))
         (amount (min base max-afford)))
    (max 0 (min amount stack))))
            
;;;; ------------------------------------------------------------------
;;;; Helper decision functions (preflop / postflop)
;;;; ------------------------------------------------------------------

(defun ai-preflop-decision (win-prob to-call stack mdp-act p-loose p-tight)
  (let ((raise-amount (choose-preflop-raise-amount win-prob stack)))
    (cond
      ;; No bet to call: decide whether to open-raise or just limp/check.
      ((= to-call 0)
       (cond
         ;; Very good preflop equity vs 3 opponents is ~0.35+.
         ((>= win-prob 0.35)
          (if (> stack raise-amount)
              raise-amount
              'in))
         ;; Decent hand vs loose opponents: sometimes raise as an iso/value bet.
         ((and (>= win-prob 0.30)
               (> p-loose p-tight)
               (> stack raise-amount)
               (> (random 100) 70))
          raise-amount)
         ;; Everything else: just see a cheap flop.
         (t 'in)))

      ;; There is a bet to call preflop.
      (t
       (cond
         ;; Very poor equity: fold.
         ((< win-prob 0.22)
          'fold)

         ;; Marginal hand, but cheap to see a flop and stack is ok.
         ((and (< win-prob 0.30)
               (<= to-call raise-amount)
               (> stack to-call))
          'in)

         ;; Reasonable/good hand: call or raise depending on MDP and villain looseness.
         ((>= win-prob 0.30)
          (cond
            ((and (> win-prob 0.38)
                  (> stack (+ to-call raise-amount))
                  (or (eq mdp-act :raise)
                      (> p-loose p-tight)))
             raise-amount)
            ((> stack to-call)
             'in)
            (t 'fold)))

         (t 'fold))))))


(defun ai-postflop-decision (win-prob to-call stack pot-size pot-odds
                                      call-margin raise-thresh mdp-act
                                      p-loose p-tight)
  (let ((raise-amount (choose-postflop-raise-amount win-prob stack pot-size)))
    (cond
      ;; No bet to call: we either check or bet for value.
      ((= to-call 0)
       (cond
         ;; Big made hand: value bet / raise.
         ((and (>= win-prob raise-thresh)
               (> stack raise-amount))
          raise-amount)

         ;; Medium+ equity vs very loose villain: occasionally stab/probe.
         ((and (>= win-prob (+ pot-odds call-margin))
               (> p-loose p-tight)
               (> stack raise-amount)
               (> pot-size 80)
               (> (random 100) 70))
          raise-amount)

         ;; Otherwise: just check.
         (t 'in)))

      ;; There is a bet to call.
      (t
       (let ((break-even (+ pot-odds call-margin)))
         (cond
           ;; Very strong hand: raise for value if we can afford it.
           ((and (>= win-prob raise-thresh)
                 (> stack (+ to-call raise-amount)))
            raise-amount)

           ;; Enough equity to profitably call.
           ((and (>= win-prob break-even)
                 (> stack to-call))
            'in)

           ;; Short-stacked: don't overfold if we still have decent equity.
           ((and (< stack 80)
                 (> win-prob 0.40)
                 (> stack to-call))
            'in)

           ;; Otherwise, fold.
           (t 'fold)))))))

(defun simple-hand-category (board my-hand)
  (let* ((cards (append board my-hand))
         (rank-counts (make-hash-table))
         (suit-counts (make-hash-table)))
    ;; Count ranks and suits
    (dolist (c cards)
      (let ((r (first c))
            (s (second c)))
        (incf (gethash r rank-counts 0))
        (incf (gethash s suit-counts 0))))
    (let ((pairs 0)
          (trips 0)
          (quads 0))
      (maphash (lambda (r cnt)
                 (declare (ignore r))
                 (cond
                   ((= cnt 4) (incf quads))
                   ((= cnt 3) (incf trips))
                   ((= cnt 2) (incf pairs))))
               rank-counts)
      (let ((max-suit-count 0))
        (maphash (lambda (s cnt)
                   (declare (ignore s))
                   (when (> cnt max-suit-count)
                     (setf max-suit-count cnt)))
                 suit-counts)
        (cond
          ((> quads 0) :monster)
          ((and (> trips 0) (or (> pairs 0) (> trips 1))) :monster) ; full house+
          ((or (> trips 0) (>= pairs 2) (>= max-suit-count 5)) :strong)
          ((= pairs 1) :medium)
          (t :air))))))
          
(defun normalize-ai-action (action)
  (cond
    ((member action '(fold in) :test #'eq)
     action)
    ((and (integerp action) (>= action 0))
     action)
    (t
     (format t "~&[AI WARNING] Invalid action ~S, forcing 'in.~%" action)
     'in)))

;;;; ------------------------------------------------------------------
;;;; Single combined AI decision function
;;;; ------------------------------------------------------------------

(defun ai-decide (state game-state)
  ;; 1) Update opponent model (HMM + stats) from betting behavior
  (update-all-opponent-beliefs game-state)
  (let* ((belief   (main-opponent-belief game-state))
         (p-tight  (getf belief :tight))
         (p-loose  (getf belief :loose))
         ;; style from stats/HMM: :nit / :station / :aggro / :balanced
         (style    (or (getf belief :style) :balanced))
         (street   (decision-state-street state))
         (to-call  (decision-state-to-call state))
         (pot-size (decision-state-pot-size state))
         (stack    (decision-state-my-stack state))
         (my-hand  (AI-get-hand))
         (board    (decision-state-cards-on-table state))
         ;; active opponents (not folded, not AI)
         (active-opps
          (remove-if (lambda (pl)
                       (or (equal (first pl) 'AI)
                           (equal (fourth pl) 'FOLD)))
                     game-state))
         (num-opps (length active-opps))
         (num-opps (if (> num-opps 0) num-opps 1)) ; at least 1
         ;; Monte Carlo sample size depends on street
         (samples (cond
                    ((eq street :preflop) 150)
                    ((eq street :flop)    120)
                    (t                    80)))
         ;; Multi-opponent equity
         (win-prob (estimate-win-probability-mc-multi
                    board my-hand num-opps :samples samples))
         ;; Pot odds: TO-CALL / (POT + TO-CALL)
         (pot-plus-call (let ((x (+ pot-size to-call)))
                          (if (> x 0) x 1)))
         (pot-odds (if (> to-call 0)
                       (/ (float to-call) pot-plus-call)
                       0.0))
         ;; MDP abstract recommendation
         (mdp-act (mdp-recommend-action state belief))

         ;; --- STYLE-BASED TUNING (extra nasty vs NOOB2-as-:station) ---

         ;; :station => basically pure pot-odds calling.
         (call-margin
          (case style
            (:nit      0.10)  ; vs nits, only continue with clear edge
            (:aggro    0.04)  ; vs aggro, okay to call somewhat looser
            (:station  0.00)  ; vs calling stations (NOOB2), call whenever +EV
            (:balanced 0.05)
            (t         0.05)))

         ;; Equity threshold to value-raise.
         ;; :station => raise thinner; they will pay us off.
         (raise-thresh
          (case style
            (:nit      0.72)
            (:aggro    0.60)
            (:station  0.52)  ; punish NOOB2: more thin value-raises
            (:balanced 0.64)
            (t         0.64)))

         action)

    ;; === Decide action by street ===
    (setf action
          (if (eq street :preflop)
              ;; Preflop logic (uses HMM tight/loose + MDP)
              (ai-preflop-decision win-prob to-call stack mdp-act p-loose p-tight)
              ;; Postflop logic (uses equity, pot odds, *style-tuned* margins)
              (ai-postflop-decision win-prob to-call stack pot-size pot-odds
                                    call-margin raise-thresh mdp-act
                                    p-loose p-tight)))

        ;; === Safety overrides ===
    (setf action
          (cond
            ((and (= to-call 0)
                  (eq action 'fold))
             'in)
            ((and (< stack 80)
                  (> win-prob 0.40)
                  (eq action 'fold))
             'in)
            (t action)))

    ;; Ensure we only ever return 'fold, 'in, or an integer
    (setf action (normalize-ai-action action))

    (log-decision :ai state action)
    action))

;;;; ------------------------------------------------------------------
;;;; Main AI entry point for the engine
;;;; ------------------------------------------------------------------

(defun AI (cards-on-table game-state)
  (handler-case
      (let ((state (extract-decision-state cards-on-table game-state)))
        (ai-decide state game-state))
    (error (e)
      ;; Optional: log the error so you can see if something still occasionally breaks
      (format t "~&[AI ERROR] ~A~%" e)
      ;; Fallback action: just check/call
      'in)))

;;;; ------------------------------------------------------------------
;;;; Helper: check if there is only one player still in the hand
;;;; ------------------------------------------------------------------

(defun last-player-in (game-state)
  (let* ((active (remove-if (lambda (x) (equal (fourth x) 'FOLD))
                            game-state))
         (num-active (length active)))
    (= num-active 1)))

;;;; ------------------------------------------------------------------
;;;; Sample players: noob1, noob2, noob3
;;;; ------------------------------------------------------------------

;;; noob1: simple player, never raises, sometimes folds
(defparameter *noob1-hand* nil)

(defun noob1-set-hand (cards)
  (setf *noob1-hand* cards))

(defun noob1-get-hand ()
  *noob1-hand*)

(defun noob1 (cards-on-table game-state)
  (declare (ignore cards-on-table))
  ;; noob1 never raises, either checks/calls or folds
  (if (last-player-in game-state)
      'in
      (if (> (random 10) 7)
          'fold
          'in)))

;;; noob2: simple player, never folds, sometimes raises fixed amount
(defparameter *noob2-hand* nil)

(defun noob2-set-hand (cards)
  (setf *noob2-hand* cards))

(defun noob2-get-hand ()
  *noob2-hand*)

(defun noob2 (cards-on-table game-state)
  (declare (ignore cards-on-table))
  (let* ((state (find-if (lambda (x) (equal (first x) 'NOOB2))
                         game-state))
         (chips (second state)))
    ;; noob2 never folds, either checks/calls or sometimes raises 40
    (if (last-player-in game-state)
        'in
        (if (and (> chips 100) (> (random 10) 7))
            40        ; raise 40 if enough chips and in the mood
            'in))))

;;; noob3: simple player, like noob1
(defparameter *noob3-hand* nil)

(defun noob3-set-hand (cards)
  (setf *noob3-hand* cards))

(defun noob3-get-hand ()
  *noob3-hand*)

(defun noob3 (cards-on-table game-state)
  (declare (ignore cards-on-table))
  ;; noob3 never raises, either checks/calls or folds
  (if (last-player-in game-state)
      'in
      (if (> (random 10) 7)
          'fold
          'in)))

(defun update-opponent-beliefs-from-showdown (output)
  (dolist (name '(NOOB1 NOOB2 NOOB3))
    (when (search (format nil "Player ~A has hand:" name) output
                  :test #'char-equal)
      (let* ((belief   (get-opponent-belief name))
             (p-tight  (getf belief :tight))
             (p-loose  (getf belief :loose))
             (new-p-loose (min 0.99 (+ p-loose 0.05)))
             (new-p-tight (- 1.0 new-p-loose)))
        (setf (gethash name *opponent-beliefs*)
              (list :tight new-p-tight :loose new-p-loose))))))

;;;; ------------------------------------------------------------------
;;;; Testing and benchmarking functions
;;;; ------------------------------------------------------------------

;;;; ------------------------------------------------------------------
;;;; Benchmark helpers
;;;; ------------------------------------------------------------------

(defun test-full-AI-with-winner ()
  "Run test-full-AI silently and return :ai, :noob1, :noob2, :noob3, or :unknown."
  (let ((output
          (with-output-to-string (s)
            ;; send ALL output only into string S, NOT to console
            (let ((*standard-output* s))
              (test-full-AI)))))
    (cond
      ((search "Game finished: AI takes all" output :test #'char-equal)
       :ai)
      ((search "Game finished: NOOB1 takes all" output :test #'char-equal)
       :noob1)
      ((search "Game finished: NOOB2 takes all" output :test #'char-equal)
       :noob2)
      ((search "Game finished: NOOB3 takes all" output :test #'char-equal)
       :noob3)
      (t
       :unknown))))

(defun benchmark-bot (label n)
  "Run N full games and report how many times each player wins.
LABEL is just a name to print in the results (e.g., :ai)."
  (let ((ai-wins    0)
        (noob1-wins 0)
        (noob2-wins 0)
        (noob3-wins 0)
        (unknown    0))
    (dotimes (i n)
      (format t "~&--- Game ~D for bot ~A ---~%" (1+ i) label)
      (clear-decision-log)
      (let ((winner (test-full-AI-with-winner)))
        ;; print this game’s winner:
        (format t "Winner of game ~D: ~A~%" (1+ i) winner)
        (case winner
          (:ai    (incf ai-wins))
          (:noob1 (incf noob1-wins))
          (:noob2 (incf noob2-wins))
          (:noob3 (incf noob3-wins))
          (t      (incf unknown)))))
    (format t "~&RESULT over ~D games (label ~A):~%" n label)
    (format t "  AI wins:    ~D~%" ai-wins)
    (format t "  NOOB1 wins: ~D~%" noob1-wins)
    (format t "  NOOB2 wins: ~D~%" noob2-wins)
    (format t "  NOOB3 wins: ~D~%" noob3-wins)
    (format t "  Unknown:    ~D~%" unknown)
    (values ai-wins noob1-wins noob2-wins noob3-wins unknown)))