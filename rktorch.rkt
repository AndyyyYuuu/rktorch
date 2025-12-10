;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "htdp-intermediate-lambda-reader.ss" "lang")((modname rktorch) (read-case-sensitive #t) (teachpacks ()) (htdp-settings #(#t constructor repeating-decimal #f #t none #f () #f)))


;; NdArr processing functions
;; ~~~~~~~~~~

;; An NdArr is one of
;; * Num
;; * (listof NdArr), with all elements of the same shape

;; Gets the shape of the NdArr
;; ndarr/shape: NdArr -> (listof Nat)
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

;; Operates on two NdArrs element-wise
;; ndarr/elementwise: (Num -> Num) NdArr NdArr -> NdArr
;; Requires: the ndarrs are the same shape
(define (ndarr/elementwise op t1 t2)
  (cond
    [(and (number? t1) (number? t2)) (op t1 t2)]
    [else (map (lambda (a b) (ndarr/elementwise op a b)) t1 t2)]))

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


(check-expect (ndarr/shape A) (list 3 3 2))

(check-expect (ndarr/elementwise + A B) (list
 (list (list 3 4) (list 8 9) (list 2 10))
 (list (list 2 10) (list 2 1) (list 4 8))
 (list (list 2 3) (list 5 4) (list 2 1))))

(define (ndarr/to-tensor ndarr)
  (mk-tensor ndarr (ndarr/fill (ndarr/shape ndarr) 0) 'none '() (lambda (x) x)))

;; Tensor functions
;; ~~~~~~~~~~

;; An Op is (anyof 'matmul 'tanh 'relu 'add 'none)

;; A Tensor is a (listof NdArr NdArr Op (listof Tensor) Tensor ())

(define (mk-tensor ndarr grad prev-op presyns backward)
    (list ndarr grad prev-op presyns backward))


(define (tensor/ndarr tensor)
  (first tensor))

(define (tensor/grad tensor)
  (second tensor))

(define (tensor/prev-op tensor)
  (third tensor))

(define (tensor/presyns tensor)
  (fourth tensor))

(define (tensor/backward tensor)
  (fifth tensor))

(define (tensor/new-grad t grad)
  (mk-tensor (tensor/ndarr t) grad (tensor/prev-op t) (tensor/presyns t) (tensor/backward t)))

(define (tensor/new-presyns t presyns)
  (mk-tensor (tensor/ndarr t) (tensor/grad t) (tensor/prev-op t) presyns (tensor/backward t)))

(define (tensor/leaf? t)
  (symbol=? 'none (tensor/prev-op t)))

(define (tensor/new-ndarr t value)
  (mk-tensor value (tensor/grad t) (tensor/prev-op t) (tensor/presyns t) (tensor/backward t)))


(define (tensor/add t1 t2)
  (mk-tensor (ndarr/elementwise + (tensor/ndarr t1) (tensor/ndarr t2))
             (ndarr/fill (ndarr/shape (tensor/ndarr t1)) 0)
             'add
             (list t1 t2)
             ;; Tensor -> (listof Tensor)
             (lambda (out-t)
               (cond
                 [(tensor/leaf? out-t) out-t]
                 [else
                  (local
                    [(define t1 (first (tensor/presyns out-t)))
                     (define t2 (second (tensor/presyns out-t)))]
                    (tensor/new-presyns out-t (list ((tensor/backward t1) (tensor/new-grad t1 (ndarr/elementwise + (tensor/grad t1) (tensor/grad out-t))))
                                                    ((tensor/backward t2) (tensor/new-grad t2 (ndarr/elementwise + (tensor/grad t2) (tensor/grad out-t)))))))]))))




;; Optimization functions
;; ~~~~~~~~~~

(define (optim/step t step-size)
  (mk-tensor (cond
               [(tensor/leaf? t) (ndarr/elementwise +
                                (tensor/ndarr t)
                                (ndarr/elementwise * (ndarr/fill (ndarr/shape (tensor/ndarr t)) (- step-size)) (tensor/grad t)))]
               [else (tensor/ndarr t)])
             (tensor/grad t)
             (tensor/prev-op t)
             (map (lambda (x) (optim/step x step-size)) (tensor/presyns t))
             (tensor/backward t)))

(define (optim/backward t)
  ((tensor/backward t) (tensor/new-grad t (ndarr/fill (ndarr/shape (tensor/ndarr t)) 1))))
             ;(lambda (t)
             ;  [(define c (tensor/presyns t))]
             ;  (list (ndarr/elementwise * (tensor/ndarr t2) (tensor/))


(define (optim/loop t n step-size)
  (cond
    [(zero? n) t]
    [else (optim/loop (optim/step (optim/backward t) step-size) (sub1 n) step-size)]))
(define toy-graph (tensor/add (ndarr/to-tensor '((1 2) (3 4))) (ndarr/to-tensor '((1 1) (1 1)))))

(optim/backward toy-graph)
'------
(optim/backward (optim/step (optim/backward toy-graph) 0.01))
'------
(optim/step (optim/backward (optim/step (optim/backward toy-graph) 0.01)) 0.01)
'------
(optim/loop toy-graph 20 0.01)

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