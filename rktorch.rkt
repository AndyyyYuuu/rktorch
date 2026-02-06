;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "htdp-advanced-reader.ss" "lang")((modname rktorch) (read-case-sensitive #t) (teachpacks ()) (htdp-settings #(#t constructor repeating-decimal #t #t none #f () #f)))
;; TRAINABLE TOY NEURAL NETWORK
;; Andy Yu

;; A few basic operations
;; ~~~~~~~~~~

;; tanh: Num -> Num
(define (tanh x)
  (/ (- 1 (exp (* -2 x))) (+ 1 (exp (* -2 x)))))

;; tanh: Num -> Num
(define (tanh-deriv x)
  (- 1 (expt (tanh x) 2)))

;; list/nth: Nat (listof X) -> X
;; Requires: the index is valid
(define (list/nth n lst)
  (cond
    [(zero? n) (first lst)]
    [else (list/nth (sub1 n) (rest lst))]))

;; list/mean: (listof Num) -> Num
(define (list/mean lst)
  (/ (foldr + 0 lst) (length lst)))

;; Random number functions
;; ~~~~~~~~~~

(define random/uniform/precision 10000)

;; random/uniform: Num Num -> Num
(define (random/uniform a b)
  (+ a (* (- b a) (/ (+ (random random/uniform/precision) 1) random/uniform/precision))))

;; random/gaussian: Num Num -> Num
(define (random/gaussian mean sd)
  (+ mean (* sd (sqrt (* -2 (log (random/uniform 0 1)))) (cos (* 2 pi (random/uniform 0 1))))))

(define noise/sd 0.03)

;; random/add-noise: Num -> Num
(define (random/add-noise n)
  (+ n (random/gaussian 0 noise/sd)))

;; NdArr processing functions
;; ~~~~~~~~~~

;; For simplicity, I think we're gonna use the
;; innermost / last dimension as the batch dimension.
;; Actually, never mind, let's not batch around. 

;; An NdArr is one of
;; * Num
;; * (listof NdArr), with all elements of the same shape

;; A Shape is a (listof Nat)

;; Gets the shape of the NdArr
;; ndarr/shape: NdArr -> Shape
(define (ndarr/shape x)
  (cond
    [(number? x) empty]
    [else (cons (length x) (ndarr/shape (first x)))]))

;; Makes a deep copy of the NdArr
;; ndarr/copy: NdArr -> NdArr
(define (ndarr/copy x)
  (cond
    [(empty? x) empty]
    [(list? x) (cons (ndarr/copy (first x)) (ndarr/copy (rest x)))]
    [else x]))

;; Creates an NdArr of size shape filled with x
;; ndarr/fill: (listof Nat) Num -> NdArr
(define (ndarr/fill shape x)
  (cond
    [(empty? shape) x]
    [else (build-list (first shape) (lambda (i) (ndarr/fill (rest shape) x)))]))

(define (ndarr/map op t)
  (cond
    [(number? t) (op t)]
    [else (map (lambda (x) (ndarr/map op x)) t)]))

;; Operates on two NdArrs element-wise
;; ndarr/elementwise: (Num -> Num) NdArr NdArr -> NdArr
;; Requires: the ndarrs are the same shape
(define (ndarr/elementwise op t1 t2)
  (cond
    [(and (number? t1) (number? t2)) (op t1 t2)]
    [else (map (lambda (a b) (ndarr/elementwise op a b)) t1 t2)]))

(define (ndarr/get idx x)
  (cond
    [(empty? idx) x]
    [else (ndarr/get (rest idx) (list/nth (first idx) x))]))

;; Builds an NdArr of the given Shape,
;; and populates it with the outputs of
;; the given function taking index as input.
;; ndarr/build: Shape (Shape -> Any) -> NdArr
(define (ndarr/build shape fn)
  (cond
    [(empty? shape) (fn empty)]
    [else (build-list (first shape)
                      (lambda (i) (ndarr/build (rest shape)
                                               (lambda (idx) (fn (cons i idx))))))]))

;; ndarr/transpose: NdArr -> NdArr
;; Requires: m is a 2d matrix
(define (ndarr/transpose m)
  (local
    [(define s (ndarr/shape m))]
    (ndarr/build (list (second s) (first s)) (lambda (i) (ndarr/get (list (second i) (first i)) m)))))

;(check-expect (ndarr/transpose '((1 2) (2 3) (3 4))) '((1 2 3) (2 3 4)))

;; Takes the dot product of two NdArrs
;; ndarr/dot: NdArr NdArr -> NdArr
;; Requires: (first (shape v)) = (first (shape w))
(define (ndarr/dot v w)
  (cond
    [(empty? (rest v)) (ndarr/elementwise * (first v) (first w))] ; Not pretty but I can't broadcast + 0
    [else (ndarr/elementwise + (ndarr/elementwise * (first v) (first w)) (ndarr/dot (rest v) (rest w)))]))

;; ndarr/mse: NdArr NdArr -> NdArr
;; Requires: v1 and v2 are vectors of equal shape
(define (ndarr/mse v1 v2)
  (ndarr/elementwise / (foldr (lambda (x y) (ndarr/elementwise + x y))
            0
            (ndarr/elementwise (lambda (a b) (ndarr/map sqr (ndarr/elementwise - a b))) v1 v2)) (length v1)))

;; ndarr/vm-mul: NdArr NdArr -> NdArr
(define (ndarr/vm-mul v m)
  (cond [(= (length v) (length (first m)))
         (map (lambda (r) (ndarr/dot v r)) m)]))

;; ndarr/outer-prod: NdArr NdArr -> NdArr
(define (ndarr/outer-prod vi vj)
  (map (lambda (r) (map (lambda (c) (ndarr/elementwise * c r)) vj)) vi))




(check-expect (ndarr/vm-mul '(1 2) '((2 3) (0 0) (1 1)))
               '(8 0 3))

;; Flattens the NdArr
;; ndarr/flatten: NdArr -> NdArr
(define (ndarr/flatten t)
  (cond [(empty? t) empty]
        [(list? (first t)) (append (ndarr/flatten (first t)) (ndarr/flatten (rest t)))]
        [else (cons (first t) (ndarr/flatten (rest t)))]))

;(check-expect (ndarr/outer-prod '(1 2) '(1 2 3)) '((1 2 3)
;                                                   (2 4 6)))
;
;(check-expect (ndarr/dot '((1 2) (3 2) (4 5)) '((2 2) (4 4) (0 1))) '(14 17))
;
;(check-expect (ndarr/build '(4 4) (lambda (x) (if (< (first x) (second x)) 0 1)))
;              '((1 0 0 0)
;                (1 1 0 0)
;                (1 1 1 0) ; look ma, causal attention mask!
;                (1 1 1 1)))
;
;(define A (list (list (list 2 3) (list 7 8) (list 1 9))
;                (list (list 1 9) (list 1 0) (list 3 7))
;                (list (list 1 2) (list 4 3) (list 1 0))))
;(define B (ndarr/fill (list 3 3 2) 1))
;

(check-expect (ndarr/vm-mul '(1 -1 2) '((1 -1 0) (3 -2 1))) '(2 7))
;
;(check-expect (ndarr/flatten A) '(2 3 7 8 1 9 1 9 1 0 3 7 1 2 4 3 1 0))
;(check-expect (ndarr/shape A) (list 3 3 2))
;
;(check-expect (ndarr/elementwise + A B) '(((3 4 ) (8 9) (2 10))
;                                          ((2 10) (2 1) (4 8 ))
;                                          ((2 3 ) (5 4) (2 1 ))))

;; Tensor functions
;; ~~~~~~~~~~

;; An Op is (anyof 'matmul 'tanh 'relu 'add)
;; A Tag is (anyof Op 'param 'X 'Y 'Y-pred)

;; A Tensor is a (listof NdArr NdArr Tag (listof Tensor) (Tensor -> Tensor) (Tensor -> (listof Tensor)))

;; tensor/mk-tensor: (listof NdArr NdArr Tag (listof Tensor) (Tensor -> Tensor) (Tensor -> (listof Tensor))) -> Tensor
(define (mk-tensor ndarr grad tag presyns forward backward)
  (cond [(and (symbol? tag) ; type checking condition
              (list? presyns)
              (or (empty? presyns)
                  (and (list? (first presyns)) (empty? (rest presyns)))
                  (and (list? (first presyns)) (list? (second presyns)))))
         (list ndarr grad tag presyns forward backward)]))

;; tensor/copy: Tensor -> Tensor
(define (tensor/copy t)
  (mk-tensor (tensor/ndarr t) (tensor/grad t) (tensor/tag t) (tensor/presyns t) (tensor/forward t) (tensor/backward t)))

;; Tensor Getters

(define (tensor/ndarr tensor)
  (first tensor))

(define (tensor/grad tensor)
  (second tensor))

(define (tensor/tag tensor)
  (third tensor))

(define (tensor/presyns tensor)
  (fourth tensor))

(define (tensor/forward tensor)
  (fifth tensor))

(define (tensor/backward tensor)
  (sixth tensor))

;; Tensor Setters

;; tensor/set-grad: Tensor NdArr -> Tensor
(define (tensor/set-grad t grad)
  (mk-tensor (tensor/ndarr t) grad (tensor/tag t) (tensor/presyns t) (tensor/forward t) (tensor/backward t)))

;; tensor/set-presyns: Tensor (listof Tensor) -> Tensor
(define (tensor/set-presyns t presyns)
  (mk-tensor (tensor/ndarr t) (tensor/grad t) (tensor/tag t) presyns (tensor/forward t) (tensor/backward t)))

;; Create a parameter
;; tensor/set-grad: Shape -> Tensor
(define (tensor/new-param shape)
  (mk-tensor (ndarr/build shape (lambda (_) (random/gaussian 0 0.2))) (ndarr/fill shape 0) 'param '() (lambda (self) (tensor/copy self)) (lambda (x) x)))

;; Create an input node
;; tensor/new-input: NdArr -> Tensor
(define (tensor/new-input ndarr)
  (mk-tensor ndarr (ndarr/fill (ndarr/shape ndarr) 0) 'X '() (lambda (self) (tensor/copy self)) (lambda (x) x)))

;; Create an expected output node
;; tensor/new-targ: NdArr -> Tensor
(define (tensor/new-targ ndarr)
  (mk-tensor ndarr (ndarr/fill (ndarr/shape ndarr) 0) 'Y '() (lambda (self) (tensor/copy self)) (lambda (x) x)))


(define (tensor/param? t)
  (symbol=? 'param (tensor/tag t)))

(define (tensor/input? t)
  (symbol=? 'X (tensor/tag t)))

(define (tensor/targ? t)
  (symbol=? 'Y (tensor/tag t)))

(define (tensor/output? t)
  (symbol=? 'Y-pred (tensor/tag t)))

(define (tensor/leaf? t)
  (or (symbol=? 'param (tensor/tag t))
      (symbol=? 'X (tensor/tag t))
      (symbol=? 'Y (tensor/tag t))))

(define (tensor/set-ndarr t value)
  (mk-tensor value (tensor/grad t) (tensor/tag t) (tensor/presyns t) (tensor/forward t) (tensor/backward t)))

(define (tensor/set-tag t value)
  (mk-tensor (tensor/ndarr t) (tensor/grad t) value (tensor/presyns t) (tensor/forward t) (tensor/backward t)))


;; Differentiable Tensor Functions
;; ~~~~~~~~~~

;; tensor/add: Tensor Tensor -> Tensor
(define (tensor/add t1 t2)
  (mk-tensor (ndarr/elementwise + (tensor/ndarr t1) (tensor/ndarr t2))
             (ndarr/fill (ndarr/shape (tensor/ndarr t1)) 0)
             'add
             (list t1 t2)
             (lambda (self)
               (local [(define t1 ((tensor/forward (first (tensor/presyns self))) (first (tensor/presyns self))))
                       (define t2 ((tensor/forward (second (tensor/presyns self))) (second (tensor/presyns self))))]
                 (mk-tensor (ndarr/elementwise + (tensor/ndarr t1) (tensor/ndarr t2))
                            (tensor/grad self)
                            'add
                            (list t1 t2)
                            (tensor/forward self)
                            (tensor/backward self))))
             ;; Tensor -> (listof Tensor) 
             (lambda (out-t)
               (cond
                 [(tensor/leaf? out-t) out-t]
                 [else
                  (local
                    [(define t1 (first (tensor/presyns out-t)))
                     (define t2 (second (tensor/presyns out-t)))]
                    (tensor/set-presyns out-t (list ((tensor/backward t1) (tensor/set-grad t1 (tensor/grad out-t)))
                                                    ((tensor/backward t2) (tensor/set-grad t2 (tensor/grad out-t))))))]))))

;; tensor/vm-mul: Tensor Tensor -> Tensor
(define (tensor/vm-mul t1 t2)
  (mk-tensor (ndarr/vm-mul (tensor/ndarr t1) (tensor/ndarr t2))
             (ndarr/fill (ndarr/shape (ndarr/vm-mul (tensor/ndarr t1) (tensor/ndarr t2))) 0)
             'vm-mul
             (list t1 t2)
             (lambda (self)
               (local [(define t1 ((tensor/forward (first (tensor/presyns self))) (first (tensor/presyns self))))
                       (define t2 ((tensor/forward (second (tensor/presyns self))) (second (tensor/presyns self))))]
                 (mk-tensor (ndarr/vm-mul (tensor/ndarr t1) (tensor/ndarr t2))
                            (tensor/grad self)
                            'vm-mul
                            (list t1 t2)
                            (tensor/forward self)
                            (tensor/backward self))))
             ;; Tensor -> (listof Tensor) 
             (lambda (out-t)
               (cond
                 [(tensor/leaf? out-t) out-t]
                 [else
                  (local
                    [(define v (first (tensor/presyns out-t)))
                     (define m (second (tensor/presyns out-t)))]
                    (tensor/set-presyns out-t (list ((tensor/backward v) (tensor/set-grad v (ndarr/vm-mul (tensor/grad out-t) (ndarr/transpose (tensor/ndarr m)))))
                                                    ((tensor/backward m) (tensor/set-grad m (ndarr/outer-prod (tensor/grad out-t) (tensor/ndarr v)))))))]))))

;; tensor/mse: Tensor Tensor -> Tensor
(define (tensor/mse y-pred y-targ)
  (mk-tensor (ndarr/mse (tensor/ndarr y-pred) (tensor/ndarr y-targ))
             (ndarr/fill (ndarr/shape (tensor/ndarr y-targ)) 0)
             'mse
             (list (tensor/set-tag y-pred 'Y-pred) y-targ)
             (lambda (self)
               (local [(define t1 ((tensor/forward (first (tensor/presyns self))) (first (tensor/presyns self))))
                       (define t2 ((tensor/forward (second (tensor/presyns self))) (second (tensor/presyns self))))]
                 (mk-tensor (ndarr/mse (tensor/ndarr t1) (tensor/ndarr t2))
                            (tensor/grad self)
                            'mse
                            (list t1 t2)
                            (tensor/forward self)
                            (tensor/backward self))))
             ;; Tensor -> (listof Tensor) 
             (lambda (out-t)
               (cond
                 [(tensor/leaf? out-t) out-t]
                 [else
                  (local
                    [(define t1 (first (tensor/presyns out-t)))
                     (define t2 (second (tensor/presyns out-t)))
                     (define n (length t1))]
                    (tensor/set-presyns out-t (list ((tensor/backward t1) (tensor/set-grad t1
                                                                                          (ndarr/map (lambda (x) (* x (/ 2 n)))
                                                                                                      (ndarr/elementwise - (tensor/ndarr t1) (tensor/ndarr t2)))))
                                                    ((tensor/backward t2) (tensor/set-grad t2
                                                                                          0)))))]))))

;; tensor/tanh: Tensor -> Tensor
(define (tensor/tanh t)
  (mk-tensor (ndarr/map tanh (tensor/ndarr t)) 
             (ndarr/fill (ndarr/shape (tensor/ndarr t)) 0)
             'tanh
             (list t)
             
             (lambda (self)
                  (local
                    [(define t ((tensor/forward (first (tensor/presyns self))) (first (tensor/presyns self))))]
                    (mk-tensor (ndarr/map tanh (tensor/ndarr t))
                            (tensor/grad self)
                            'tanh
                            (list t)
                            (tensor/forward self)
                            (tensor/backward self))))
             ;; Tensor -> (listof Tensor)
             (lambda (out-t)
               (cond
                 [(tensor/leaf? out-t) out-t]
                 [else
                  (local
                    [(define t (first (tensor/presyns out-t)))]
                    (tensor/set-presyns out-t (list ((tensor/backward t)
                                                     (tensor/set-grad t (ndarr/elementwise * (ndarr/map tanh-deriv (tensor/ndarr t)) (tensor/grad out-t)))))))]))))



;; Optimization functions
;; ~~~~~~~~~~

;; optim/step: Tensor Num -> Tensor
(define (optim/step t step-size)
  (mk-tensor (cond
               [(tensor/param? t) (ndarr/elementwise +
                                (tensor/ndarr t)
                                (ndarr/elementwise * (ndarr/fill (ndarr/shape (tensor/ndarr t)) (- step-size)) (tensor/grad t)))]
               [else (tensor/ndarr t)])
             (tensor/grad t)
             (tensor/tag t)
             (map (lambda (x) (optim/step x step-size)) (tensor/presyns t))
             (tensor/forward t)
             (tensor/backward t)))

;; optim/backward: Tensor -> Tensor
(define (optim/backward t)
  ((tensor/backward t) (tensor/set-grad t (ndarr/fill (ndarr/shape (tensor/ndarr t)) 1))))

;; optim/replace-value: Tensor (Tensor -> Bool) NdArr -> Tensor
(define (optim/replace-value t pred? value)
       (cond
         [(pred? t) (tensor/set-ndarr t value)]
         [else (tensor/set-presyns t (map (lambda (x) (optim/replace-value x pred? value)) (tensor/presyns t)))]))

;; optim/forward: Tensor NdArr -> Tensor
(define (optim/forward t input target)
  ((tensor/forward t) (optim/replace-value (optim/replace-value t tensor/input? input) tensor/targ? target)))

;; Runs one epoch of training
;; optim/epoch: Tensor Num Dataset empty -> (list Tensor (listof Num))
(define (optim/epoch t step-size dataset loss-acc)
  (cond
    [(empty? dataset) (list t loss-acc)]
    [else
     (local
       [(define X (first (first dataset)))
        (define Y (second (first dataset)))
        (define updated (optim/step (optim/backward (optim/forward t X Y)) step-size))]
       
     (optim/epoch updated step-size (rest dataset) (cons (tensor/ndarr updated) loss-acc)))]))

;; Evaluates the model + loss function of the Tensor on the given dataset
;;   and produces the mean loss
;; optim/eval: Tensor Dataset -> Num
(define (optim/eval t test-set)
  (local
    [(define (loop test-set loss-acc)
       (cond
         [(empty? test-set) (list/mean loss-acc)]
         [else (local
                 [(define X (first (first test-set)))
                  (define Y (second (first test-set)))]
                 (loop (rest test-set)
                       (cons (tensor/ndarr (optim/forward t X Y)) loss-acc)))]))]
  (loop test-set empty)))

(define log-every-n-epochs 25)

;; Runs multiple epochs of training and evaluation, producing a list of:
;; - The resulting model
;; - A list of mean train losses for every nth epoch
;; - A list of mean validation losses for every nth epoch
;; optim/epochs: Nat Tensor Num (listof (list NdArr NdArr)) (listof (list NdArr NdArr)) empty -> (list Tensor (listof Num))
(define (optim/epochs n t step-size train-set test-set train-loss-acc test-loss-acc)
  (cond
    [(zero? n) (list t train-loss-acc test-loss-acc)]
    [else (local
            [(define out (optim/epoch t step-size train-set empty))
             (define new-t (first out))
             (define train-losses (second out))]
            (cond
              [(zero? (remainder n log-every-n-epochs)) (optim/epochs (sub1 n) new-t step-size train-set test-set
                                                            (cons (list/mean train-losses) train-loss-acc)
                                                            (cons (optim/eval t test-set) test-loss-acc))]
              [else (optim/epochs (sub1 n) new-t step-size train-set test-set
                                                            train-loss-acc
                                                            test-loss-acc)]))]))


;; Data synthesis functions
;; ~~~~~~~~~~

;; A DataPair is a (list NdArr NdArr)

;; A DataSet is a (listof DataBatch)

;; data/synthesize-item: _ -> DataPair
(define (data/synthesize-item _)
  (local
    [(define X_x (random/uniform -1 1))
     (define X_y (random/uniform -1 1))
     (define Y (cond
                 ;; VERY non-linear
                 [(and (not (zero? X_x)) (< (/ X_y X_x) (tan (sqrt (+ (sqr (* pi X_y)) (sqr (* pi X_x))))))) 1]
                 [else -1]))]
    (list (list (random/add-noise X_x) (random/add-noise X_y)) (list Y))))

;; data/synthesize: Nat -> DataSet
(define (data/synthesize n)
  (build-list n (lambda (_) (data/synthesize-item 0))))


;; nn/linear: Tensor Nat -> Tensor
(define (nn/linear inp-t out-size)
  (tensor/add (tensor/vm-mul inp-t
                             (tensor/new-param (list out-size
                                                     (first (ndarr/shape (tensor/ndarr inp-t))))))
              (tensor/new-param (list out-size))))

;; nn/mlp: (listof Nat) (Tensor -> Tensor) Tensor -> Tensor
(define (nn/mlp sizes act_fn input)
  (cond
    [(empty? sizes) input]
    [else (nn/mlp (rest sizes) act_fn (act_fn (nn/linear input (first sizes))))]))


(define train-set (data/synthesize 160))
(define test-set (data/synthesize 40))

(define model (nn/mlp '(4 3 1) tensor/tanh (tensor/new-input '(0 0))))
(define loss (tensor/mse model (tensor/new-targ '(0))))

'Training...

(define results (optim/epochs 256 loss 0.3 train-set test-set empty empty))

'==============================================================
"New model + loss function"
'-----------------
(first results)

'==============================================================
"Train loss (in reverse order)"
'-----------------
(second results)

'==============================================================
"Validation loss (in reverse order)"
'-----------------
(third results)