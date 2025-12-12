;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "htdp-intermediate-lambda-reader.ss" "lang")((modname rktorch) (read-case-sensitive #t) (teachpacks ()) (htdp-settings #(#t quasiquote repeating-decimal #f #t none #f () #f)))

(define (tanh x)
  (/ (- 1 (exp (* -2 x))) (+ 1 (exp (* -2 x)))))

(define (tanh-deriv x)
  (- 1 (expt (tanh x) 2)))

(define (list/nth n lst)
  (cond
    [(zero? n) (first lst)]
    [else (list/nth (sub1 n) (rest lst))]))

;; NdArr processing functions
;; ~~~~~~~~~~

;; For simplicity, I think we're gonna use the
;; innermost / last dimension as the batch dimension.

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


(define (ndarr/transpose m)
  (local
    [(define s (ndarr/shape m))]
    (ndarr/build (list (second s) (first s)) (lambda (i) (ndarr/get (list (second i) (first i)) m)))))

;(check-expect (ndarr/transpose '((1 2) (2 3) (3 4))) '((1 2 3) (2 3 4)))

;; Takes the dot product of two NdArrs
;; ndarr/dot: NdArr NdArr -> Ndarr
;; Requires: (first (shape v)) = (first (shape w))
(define (ndarr/dot v w)
  (cond
    [(empty? (rest v)) (ndarr/elementwise * (first v) (first w))] ; Not pretty but I can't broadcast + 0
    [else (ndarr/elementwise + (ndarr/elementwise * (first v) (first w)) (ndarr/dot (rest v) (rest w)))]))

(define (ndarr/mse v1 v2)
  (ndarr/elementwise / (foldr (lambda (x y) (ndarr/elementwise + x y))
            0
            (ndarr/elementwise (lambda (a b) (ndarr/map sqr (ndarr/elementwise - a b))) v1 v2)) (length v1)))

(define (ndarr/vm-mul v m)
  (cond [(= (length v) (length (first m)))
         (map (lambda (r) (ndarr/dot v r)) m)]))

(define (ndarr/outer-prod vi vj)
  (map (lambda (r) (map (lambda (c) (ndarr/elementwise * c r)) vj)) vi))






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

;; An Op is (anyof 'matmul 'tanh 'relu 'add 'none)

;; A Tensor is a (listof NdArr NdArr Op (listof Tensor) Tensor ())

(define (mk-tensor ndarr grad prev-op presyns forward backward)
  (cond [(and (symbol? prev-op) (list? presyns) (or (empty? presyns) (and (list? (first presyns)) (list? (second presyns)))))
         (list ndarr grad prev-op presyns forward backward)]))

(define (tensor/copy t)
  (mk-tensor (tensor/ndarr t) (tensor/grad t) (tensor/prev-op t) (tensor/presyns t) (tensor/forward t) (tensor/backward t)))


(define (tensor/ndarr tensor)
  (first tensor))

(define (tensor/grad tensor)
  (second tensor))

(define (tensor/prev-op tensor)
  (third tensor))

(define (tensor/presyns tensor)
  (fourth tensor))

(define (tensor/forward tensor)
  (fifth tensor))

(define (tensor/backward tensor)
  (sixth tensor))

(define (tensor/new-grad t grad)
  (mk-tensor (tensor/ndarr t) grad (tensor/prev-op t) (tensor/presyns t) (tensor/forward t) (tensor/backward t)))

(define (tensor/new-presyns t presyns)
  (mk-tensor (tensor/ndarr t) (tensor/grad t) (tensor/prev-op t) presyns (tensor/forward t) (tensor/backward t)))

;; Leaf node construction
(define (tensor/new-param ndarr)
  (mk-tensor ndarr (ndarr/fill (ndarr/shape ndarr) 0) 'param '() (lambda (self) (tensor/copy self)) (lambda (x) x)))

(define (tensor/new-input ndarr)
  (mk-tensor ndarr (ndarr/fill (ndarr/shape ndarr) 0) 'X '() (lambda (self) (tensor/copy self)) (lambda (x) x)))

(define (tensor/new-targ ndarr)
  (mk-tensor ndarr (ndarr/fill (ndarr/shape ndarr) 0) 'Y '() (lambda (self) (tensor/copy self)) (lambda (x) x)))


(define (tensor/param? t)
  (symbol=? 'param (tensor/prev-op t)))

(define (tensor/input? t)
  (symbol=? 'X (tensor/prev-op t)))

(define (tensor/targ? t)
  (symbol=? 'Y (tensor/prev-op t)))

(define (tensor/output? t)
  (symbol=? 'Y-pred (tensor/prev-op t)))

(define (tensor/leaf? t)
  (or (symbol=? 'param (tensor/prev-op t))
      (symbol=? 'X (tensor/prev-op t))
      (symbol=? 'Y (tensor/prev-op t))))

(define (tensor/new-ndarr t value)
  (mk-tensor value (tensor/grad t) (tensor/prev-op t) (tensor/presyns t) (tensor/forward t) (tensor/backward t)))

(define (tensor/flag t value)
  (mk-tensor (tensor/ndarr t) (tensor/grad t) value (tensor/presyns t) (tensor/forward t) (tensor/backward t)))

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
                    (tensor/new-presyns out-t (list ((tensor/backward t1) (tensor/new-grad t1 (tensor/grad out-t)))
                                                    ((tensor/backward t2) (tensor/new-grad t2 (tensor/grad out-t))))))]))))

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
                    (tensor/new-presyns out-t (list ((tensor/backward v) (tensor/new-grad v (ndarr/vm-mul (tensor/grad out-t) (ndarr/transpose (tensor/ndarr m)))))
                                                    ((tensor/backward m) (tensor/new-grad m (ndarr/outer-prod (tensor/grad out-t) (tensor/ndarr v)))))))]))))

(define (tensor/mse y-pred y-targ)
  (mk-tensor (ndarr/mse (tensor/ndarr y-pred) (tensor/ndarr y-targ))
             (ndarr/fill (ndarr/shape (tensor/ndarr y-targ)) 0)
             'mse
             (list (tensor/flag y-pred 'Y-pred) y-targ)
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
                    (tensor/new-presyns out-t (list ((tensor/backward t1) (tensor/new-grad t1
                                                                                          (ndarr/map (lambda (x) (* x (/ 2 n)))
                                                                                                      (ndarr/elementwise - (tensor/ndarr t1) (tensor/ndarr t2)))))
                                                    ((tensor/backward t2) (tensor/new-grad t2
                                                                                          0)))))]))))


;((tensor/backward t1) (tensor/new-grad t1 (ndarr/elementwise + (tensor/grad t1) (tensor/grad out-t))))
;((tensor/backward t2) (tensor/new-grad t2 (ndarr/elementwise + (tensor/grad t2) (tensor/grad out-t)))))))]))))

;((tensor/backward t1) (tensor/new-grad t1 (tensor/grad out-t)))
;((tensor/backward t2) (tensor/new-grad t2 (tensor/grad out-t))))))]))))
;(define (tensor/tanh t)
;  (mk-tensor (ndarr/map tanh (tensor/ndarr t)) 
;             (ndarr/fill (ndarr/shape (tensor/ndarr t)) 0)
;             'tanh
;             (list t)
;             ;; Tensor -> (listof Tensor)
;             (lambda (out-t)
;               (cond
;                 [(tensor/leaf? out-t) out-t]
;                 [else
;                  (local
;                    [(define t (first (tensor/presyns out-t)))]
;                    (tensor/new-presyns out-t (list ((tensor/backward t) (tensor/new-grad t (ndarr/map tanh-deriv (tensor/grad out-t)))))))]))))
;
;
;(define (tensor/mul t1 t2)
;  (mk-tensor (ndarr/elementwise + (tensor/ndarr t1) (tensor/ndarr t2))
;             (ndarr/fill (ndarr/shape (tensor/ndarr t1)) 0)
;             'add
;             (list t1 t2)
;             ;; Tensor -> (listof Tensor)
;             (lambda (out-t)
;               (cond
;                 [(tensor/leaf? out-t) out-t]
;                 [else
;                  (local
;                    [(define t1 (first (tensor/presyns out-t)))
;                     (define t2 (second (tensor/presyns out-t)))]
;                    (tensor/new-presyns out-t (list ((tensor/backward t1) (tensor/new-grad t1 (ndarr/elementwise * (tensor/grad t2) (tensor/grad out-t))))
;                                                    ((tensor/backward t2) (tensor/new-grad t2 (ndarr/elementwise * (tensor/grad t1) (tensor/grad out-t)))))))]))))


;; Optimization functions
;; ~~~~~~~~~~


(define (optim/step t step-size)
  (mk-tensor (cond
               [(tensor/param? t) (ndarr/elementwise +
                                (tensor/ndarr t)
                                (ndarr/elementwise * (ndarr/fill (ndarr/shape (tensor/ndarr t)) (- step-size)) (tensor/grad t)))]
               [else (tensor/ndarr t)])
             (tensor/grad t)
             (tensor/prev-op t)
             (map (lambda (x) (optim/step x step-size)) (tensor/presyns t))
             (tensor/forward t)
             (tensor/backward t)))

(define (optim/backward t)
  ((tensor/backward t) (tensor/new-grad t (ndarr/fill (ndarr/shape (tensor/ndarr t)) 1))))

(define (optim/replace-value t pred? value)
       (cond
         [(pred? t) (tensor/new-ndarr t value)]
         [else (tensor/new-presyns t (map (lambda (x) (optim/replace-value x pred? value)) (tensor/presyns t)))]))

;; optim/forward: Tensor NdArr -> Tensor
(define (optim/forward t input target)
  ((tensor/forward t) (optim/replace-value (optim/replace-value t tensor/input? input) tensor/targ? target)))


(define (optim/epoch t step-size dataset loss-acc)
  (cond
    [(empty? dataset) (list t loss-acc)]
    [else
     (local
       [(define X (first (first dataset)))
        (define Y (second (first dataset)))
        (define updated (optim/step (optim/backward (optim/forward t X Y)) step-size))]
       
     (optim/epoch updated step-size (rest dataset) (cons (tensor/ndarr updated) loss-acc)))]))

;(define toy-graph (tensor/add (tensor/new-input '((0.1 -0.3) (0.5 0.3))) (tensor/new-param '((0.3 0.1) (-0.2 0.0)))))
;(tensor/vm-mul (tensor/new-input '(1 1)) (tensor/new-param '((0.1 0.2) (0.1 0.1) (0.1 0.1))))
(define net (tensor/mse (tensor/vm-mul (tensor/new-input '(1 2)) (tensor/new-param '((0.1 0.2) (0.1 0.1) (0.1 0.1))))
                        (tensor/new-targ '(-1 -1 -1))))
;net
;
;
(optim/forward net '(1 1) '(0 0 0))
'------------------------------------------------
(optim/backward (optim/forward net '(1 1) '(0 0 0)))
'------------------------------------------------
(optim/step (optim/backward (optim/forward net '(1 1) '(0 0 0))) 0.01)
'------------------------------------------------
;(optim/backward (optim/step (optim/backward (optim/forward net '(1 1) '(0 0 0))) 0.01))
'------------------------------------------------
(optim/forward (optim/step (optim/backward (optim/forward net '(1 1) '(0 0 0))) 0.01) '(1 1) '(0 0 0))
'------------------------------------------------
;(optim/step (optim/backward (optim/step (optim/backward toy-graph) 0.01)) 0.01)
;'------------------------------------------------


(define random/uniform/precision 10000)

(define (random/uniform a b)
  (+ a (* (- b a) (/ (+ (random random/uniform/precision) 1) random/uniform/precision))))

(define (random/gaussian mean sd)
  (+ mean (* sd (sqrt (* -2 (log (random/uniform 0 1)))) (cos (* 2 pi (random/uniform 0 1))))))

(define noise/sd 0.03)

(define (random/add-noise n)
  (+ n (random/gaussian 0 noise/sd)))

(define (data/synthesize-batch n)
  (local
    [(define (loop n acc-xx acc-xy acc-y)
       (cond
         [(zero? n) (list acc-x acc-y)]
         [else (local
                 [(define X_x (random/uniform -1 1))
                  (define X_y (random/uniform -1 1))
                  (define Y (cond
                              ;; VERY non-linear
                              [(and (not (zero? X_x)) (< (/ X_y X_x) (tan (sqrt (+ (sqr (* pi X_y)) (sqr (* pi X_x))))))) 1]
                              [else -1]))]
                 (loop (sub1 n)
                       (cons (random/add-noise X_x) acc-xx)
                       (cons (random/add-noise X_y) acc-xy)
                       (cons Y acc-y)))]))]
    (loop n empty empty empty)))

(define (data/synthesize n)
  (build-list n (lambda (_) (data/synthesize-batch 1))))
(define train-set (data/synthesize 10))
;(define test-set (data/synthesize 10))
train-set
(define model (tensor/mse (tensor/vm-mul (tensor/new-input '(0 0)) (tensor/new-param '((0.1 0.2))))
                        (tensor/new-targ 0))) ; we're gonna need broadcasting for this kind of batch processing
model
;(optim/forward model '(#i0.8149676128848764
;     #i0.5999207590704589
;     #i0.8442857419868902
;     #i0.7682431672814628
;     #i-0.9443673067317068
;     #i0.7887930213363239
;     #i0.4116878432048962
;     #i-0.43814750294244836) '(1 -1 -1 -1 1 1 -1 -1))
(optim/epoch model 0.03 train-set empty)