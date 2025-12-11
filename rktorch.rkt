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
;; Takes the dot product of two NdArrs
;; ndarr/dot: NdArr NdArr -> Ndarr
;; Requires: (first (shape v)) = (first (shape w))
(define (ndarr/dot v w)
  (cond
    [(empty? (rest v)) (ndarr/elementwise * (first v) (first w))] ; Not pretty but I can't broadcast + 0
    [else (ndarr/elementwise + (ndarr/elementwise * (first v) (first w)) (ndarr/dot (rest v) (rest w)))]))

(define (ndarr/vm-mul v m^T)
  (map (lambda (r) (ndarr/dot v r)) m^T))

(check-expect (ndarr/dot '((1 2) (3 2) (4 5)) '((2 2) (4 4) (0 1))) '(14 17))

(check-expect (ndarr/build '(4 4) (lambda (x) (if (< (first x) (second x)) 0 1)))
              '((1 0 0 0)
                (1 1 0 0)
                (1 1 1 0) ; look ma, causal attention mask!
                (1 1 1 1)))

(check-expect (ndarr/vm-mul '(1 -1 2) '((1 -1 0) (3 -2 1))) '(2 7))


;; Flattens the NdArr
;; ndarr/flatten: NdArr -> NdArr
(define (ndarr/flatten t)
  (cond [(empty? t) empty]
        [(list? (first t)) (append (ndarr/flatten (first t)) (ndarr/flatten (rest t)))]
        [else (cons (first t) (ndarr/flatten (rest t)))]))

(define A (list (list (list 2 3) (list 7 8) (list 1 9))
                (list (list 1 9) (list 1 0) (list 3 7))
                (list (list 1 2) (list 4 3) (list 1 0))))
(define B (ndarr/fill (list 3 3 2) 1))

(check-expect (ndarr/flatten A) '(2 3 7 8 1 9 1 9 1 0 3 7 1 2 4 3 1 0))
(check-expect (ndarr/shape A) (list 3 3 2))

(check-expect (ndarr/elementwise + A B) '(((3 4 ) (8 9) (2 10))
                                          ((2 10) (2 1) (4 8 ))
                                          ((2 3 ) (5 4) (2 1 ))))

;; Leaf node construction
(define (tensor/new-param ndarr)
  (mk-tensor ndarr (ndarr/fill (ndarr/shape ndarr) 0) 'param '() (lambda (self) (tensor/copy self)) (lambda (x) x)))

;; 
(define (tensor/new-input ndarr)
  (mk-tensor ndarr (ndarr/fill (ndarr/shape ndarr) 0) 'input '() (lambda (self) (tensor/copy self)) (lambda (x) x)))

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

(define (tensor/param? t)
  (symbol=? 'param (tensor/prev-op t)))

(define (tensor/input? t)
  (symbol=? 'input (tensor/prev-op t)))

(define (tensor/leaf? t)
  (or (symbol=? 'param (tensor/prev-op t))
      (symbol=? 'input (tensor/prev-op t))))

(define (tensor/new-ndarr t value)
  (mk-tensor value (tensor/grad t) (tensor/prev-op t) (tensor/presyns t) (tensor/forward t) (tensor/backward t)))


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

(define optim/loss (lambda (y y-pred) y))

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

;; optim/forward: Tensor NdArr -> Tensor
(define (optim/forward t input)
  (local
    [(define (replace-input t value)
       (cond
         [(tensor/input? t) (tensor/new-ndarr t value)]
         [else (tensor/new-presyns t (map (lambda (x) (replace-input x value)) (tensor/presyns t)))]))]
  ((tensor/forward t) (replace-input t input))))


(define (optim/epoch t step-size dataset)
  (cond
    [(empty? dataset) t]
    [else
     (local
       [(define X (first (first dataset)))
        (define y (second (first dataset)))]
     (optim/epoch (optim/step (optim/backward (optim/loss (optim/forward t X) y)) step-size) step-size (rest dataset)))]))

(define toy-graph (tensor/add (tensor/new-input '((0.1 -0.3) (0.5 0.3))) (tensor/new-param '((0.3 0.1) (-0.2 0.0)))))

(optim/backward toy-graph)
'------------------------------------------------
(optim/backward (optim/step (optim/backward toy-graph) 0.01))
'------------------------------------------------
(optim/forward (optim/backward (optim/step (optim/backward toy-graph) 0.01)) '((0 0) (0 0)))
'------------------------------------------------
;(optim/step (optim/backward (optim/step (optim/backward toy-graph) 0.01)) 0.01)
;'------------------------------------------------
;(optim/loop toy-graph 100 0.1)

(define random/uniform/precision 1000)

(define (random/uniform a b)
  (+ a (* (- b a) (/ (random random/uniform/precision) random/uniform/precision))))

(define noise/max-drift 0.05)

(define (random/add-noise n)
  (+ n
     (* 1/2 (random/uniform (- noise/max-drift) noise/max-drift))
     (* 1/2 (random/uniform (- noise/max-drift) noise/max-drift))))

(define (data/synthesize n)
  (local
    [(define (loop n acc)
       (cond
         [(zero? n) acc]
         [else (loop (sub1 n) (cons (local
                                      [(define X_x (random/uniform -1 1))
                                       (define X_y (random/uniform -1 1))
                                       (define Y (cond
                                                   ;; VERY non-linear
                                                   [(< (/ X_y X_x) (tan (sqrt (+ (sqr (* pi X_y)) (sqr (* pi X_x)))))) 1]
                                                   [else -1]))]
                                      (list (list (random/add-noise X_x) (random/add-noise X_y)) Y)) acc))]))]
    (loop n empty)))
;(define train-set (data/synthesize 900))
;(define test-set (data/synthesize 100))
