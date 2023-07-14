#!/usr/bin/env racket

#lang typed/racket

#|
(require/typed math/matrix
  [#:opaque Matrix matrix?]
  [matrix+ (-> (Matrix Number) (Matrix Number) (Matrix Number))]
  [matrix- (-> (Matrix Number) (Matrix Number) (Matrix Number))]
  [matrix* (-> (Matrix Number) (Matrix Number) (Matrix Number))]
  [matrix-map (-> (-> Number Number) (Matrix Number) (Matrix Number))]
  [make-matrix (-> Integer Integer Number (Matrix Number))]
  [list->matrix (-> Integer Integer (Listof Number) (Matrix Number))]
  [matrix->list (-> (Matrix Number) (Listof Number))]
  [matrix-num-rows (-> (Matrix Number) Index)]
  [matrix-num-cols (-> (Matrix Number) Index)]
  [matrix-set-col (-> (Matrix Number) Integer (Matrix Number) (Matrix Number))]
  [matrix-set-row (-> (Matrix Number) Integer (Matrix Number) (Matrix Number))]
  [matrix-row (-> (Matrix Number) Integer (Matrix Number))]
  [matrix-col (-> (Matrix Number) Integer (Matrix Number))]
  [matrix-scale (-> (Matrix Number) Number (Matrix Number))] 
  )
|#

(require math/matrix)
(provide train
         perform
         learn
         make-nn
         print-nn
         cost)

(define-type Mat (Matrix Real))
(define-type Weight Real)
(define-type Bias Real)

(struct neural-network
  (
   [wl : (Listof Mat)]
   [bl : (Listof Mat)]))

(: print-nn (-> neural-network Void))
(define (print-nn nn)
  (begin 
    (printf "\nNN Weights:\n")
    (map mat-print (neural-network-wl nn))

    (printf "\nNN Biases:\n")
    (map mat-print (neural-network-bl nn))
    (printf "\n")
    (void)))

(struct backprop-nn
  (
   [wmat-list : (Listof (Matrix Weight))]
   [bmat-list : (Listof (Matrix Bias))]
   ))

(struct backprop-layer
  (
   [pd-prev : (Listof Real)]
   [w-mat : (Matrix Weight)]
   [b-mat : (Matrix Bias)]
   ))

(struct backprop-neuron
  (
   [w-list : (Listof Weight)]
   [pd-prev : (Listof Real)]
   ))

(: sigmoid (-> Real Real))
(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(: nn-apply (-> (-> Mat Mat Mat) neural-network neural-network neural-network))
(define (nn-apply proc nn1 nn2)
  (neural-network
   (map proc
        (neural-network-wl nn1)
        (neural-network-wl nn2))
    (map proc
        (neural-network-bl nn1)
        (neural-network-bl nn2))))

(: mat-print (-> Mat Void))
(define (mat-print mat)

  (: mat-print-rec (-> (Listof Mat) Void))
  (define (mat-print-rec rows)
    (match rows
      ['() (void)]
      [_
       (printf "\t~a\n" (car rows))
       (mat-print-rec (cdr rows))
       ]
      ))
  
  (let ((rows (matrix-rows mat)))
    (mat-print-rec rows)))
    
 
(: forward-layer (-> Mat Mat Mat Mat))
(define (forward-layer in w b)
  (matrix-map sigmoid (matrix+ (matrix* in w) b)))

(: forward (-> Mat (Listof Mat) (Listof Mat) (Listof Mat)))
(define (forward in wl bl)

  (: forward-acc (-> Mat (Listof Mat) (Listof Mat) (Listof Mat) (Listof Mat)))
  (define (forward-acc in wl bl a-acc)
    (match wl
        ['() a-acc]
        [l
         (let ((a (forward-layer in (car wl) (car bl))))
           (forward-acc a (cdr wl) (cdr bl) (cons a a-acc)))
        ]
        ))

  (forward-acc in wl bl (list in)))


(: nn-forward (-> Mat neural-network (Listof Mat)))
(define (nn-forward in nn)
  (forward in
           (neural-network-wl nn)
           (neural-network-bl nn)))

(: randf (-> Real Real))
(define (randf r)
  (- (* (random) 2) 1)
  ;; 0
  )

(: make-random-mat (-> Integer Integer Mat))
(define (make-random-mat row col)
  (matrix-map randf
              (make-matrix row col 0)
              ))

(: make-nn (-> (Listof Integer) neural-network))
(define (make-nn arch)
  (: make-wl-rec (-> (Listof Integer) (Listof Mat) (Listof Mat)))
  (define (make-wl-rec n-arch wl)
    (match n-arch
      [(list a) wl]
      [n
       (make-wl-rec
        (cdr n-arch)
        (cons (make-random-mat (car (cdr n-arch)) (car n-arch)) wl))
       ]
      ))
  
  (: make-bl-rec (-> (Listof Integer) (Listof Mat) (Listof Mat)))
  (define (make-bl-rec n-arch bl)
    (match n-arch
      [(list a) bl]
      [n
       (make-bl-rec
        (cdr n-arch)
        (cons (make-random-mat 1 (car n-arch)) bl))
       ]
      ))
  
  (let ((rev-arch (reverse arch)))
  (neural-network (make-wl-rec rev-arch '()) (make-bl-rec rev-arch '()))))

(: list-back->mat (-> Mat (Listof Real) Mat))
(define (list-back->mat mat ls) 
  (list->matrix (matrix-num-rows mat) (matrix-num-cols mat) ls))

(: mat-size (-> Mat Integer))
(define (mat-size m)
  (* (matrix-num-cols m) (matrix-num-rows m)))

(: nn-map (-> (-> Mat Mat)
              neural-network neural-network))
(define (nn-map proc nn)
    (neural-network
     (map proc (neural-network-wl nn))
     (map proc (neural-network-bl nn))))

(: cost (-> neural-network (Listof Mat) (Listof Mat) Real))
(define (cost nn input output)

  (: cost-rec (-> neural-network (Listof Mat) (Listof Mat) Real Real))
  (define (cost-rec nn input output err)
  (match input
    ['() err]
    [l
     (let* ((nn-y (car (nn-forward (car input) nn)))
           (y (car output))
           (diff (apply + (matrix->list (matrix- nn-y y)))))
       (cost-rec nn (cdr input) (cdr output) (+ err (* diff diff))))
     ]
    ))

  (/ (cost-rec nn input output 0) (length input)))

(: dcost-neuron
 (-> (Listof Real) Real Real (Listof Real) (Listof Real) Integer
     backprop-neuron))
(define (dcost-neuron w-acc-l diff-i ai ai-prev-l
                      pd-prev-list-acc i)
  (match i
    [-1 (backprop-neuron w-acc-l pd-prev-list-acc)]
    [_
     (let* ((ai-prev (list-ref ai-prev-l i))
           (wi-pd (* 2 diff-i ai (- 1 ai) ai-prev))
           (last-pd-ai-prev (list-ref pd-prev-list-acc i))
           (cur-w (list-ref w-acc-l i))
           (pd-ai-prev (* 2 diff-i ai (- 1 ai) cur-w))
           )
       
       (begin
         ;; (printf "Diff ~a, \nai ~a, \nai-prev ~a \nwi ~a\n" diff-i ai ai-prev wi-pd)
       (dcost-neuron
        (list-set w-acc-l i wi-pd)
        diff-i
        ai
        ai-prev-l
        (list-set pd-prev-list-acc i
                  (+ pd-ai-prev last-pd-ai-prev))
        (- i 1))))
     ]
    ))


;; (define (row-list->mat row-list)
  ;; (foldl 

(: dcost-layer
   (-> Integer Integer Integer (Matrix Weight) (Listof Real) (Listof Real) (Listof Real)
       (Listof Real) (Listof Real) backprop-layer))
(define (dcost-layer w-row i w-col w-mat b-acc-list next-diff-l ai-l
                     diff-l ai-prev-l)
  (match ai-l
    ['() (backprop-layer
          next-diff-l
          w-mat
          (list->matrix 1 w-col b-acc-list)
          )]
    [_
     
     (let* ((ai (car ai-l))
            (diff (car diff-l))
            (cur-row-l (matrix->list (matrix-col w-mat i)))
            (neuron-bp
             (dcost-neuron cur-row-l diff ai ai-prev-l
                           next-diff-l
                           (- w-row 1)))
            (neuron-dcost (backprop-neuron-w-list neuron-bp))
            (neuron-diff-l (backprop-neuron-pd-prev neuron-bp))
            (n-dcost-mat (list->matrix w-row 1 neuron-dcost))
            (bias-gd (* 2 diff ai (- 1 ai))))
       
       (begin
         ;; (printf "Current next: ~a\n" next-diff-l)
         ;; (printf "Current activation: ~a\n" neuron-diff-l)
         ;; (printf "Layer current col: ~a\n" i)
         (dcost-layer w-row (+ i 1) w-col
                      (matrix-set-col w-mat i n-dcost-mat)
                      (cons bias-gd b-acc-list)
                      neuron-diff-l
                      (cdr ai-l) (cdr diff-l) ai-prev-l)))
       ]
    ))

(: dcost-nn
   (-> (Listof Mat) (Listof Mat) (Listof Mat) (Listof Mat) (Listof Real) neural-network))
(define (dcost-nn forward-list wmat-list
                  wgrad-mat-list-acc bgrad-mat-list-acc diff-l)
  (match forward-list
    [(list a)
     (neural-network
      wgrad-mat-list-acc
      bgrad-mat-list-acc)
      ]
    [l
     (let* (
            (cur-wmat (car wmat-list))
            (w-rows (matrix-num-rows cur-wmat))
            (w-cols (matrix-num-cols cur-wmat))
            (layer-bp
             (dcost-layer w-rows 0 w-cols cur-wmat
                          '()
                          (make-list w-rows 0)
                          (matrix->list (car forward-list))
                          diff-l
                          (matrix->list (cadr forward-list))))
            (bp-next-diff-l (backprop-layer-pd-prev layer-bp))
            (bp-wgrad-mat (backprop-layer-w-mat layer-bp))
            (bp-bgrad-mat (backprop-layer-b-mat layer-bp))
            )
       
       (begin
         ;; (printf "BP grad w\n")
         ;; (mat-print bp-wgrad-mat)
         ;; (printf "BP grad b\n")
         ;; (mat-print bp-bgrad-mat)
         ;; (printf "\n")
         ;; (printf "Cur fwd list~a\n" (car forward-list))
         (dcost-nn (cdr forward-list) (cdr wmat-list)
                   (cons bp-wgrad-mat wgrad-mat-list-acc)
                   (cons bp-bgrad-mat bgrad-mat-list-acc)
                   (map (lambda ([d : Real]) : Real (* d 1)) bp-next-diff-l))))
     ]
    ))

(: make-zero-mat-list (-> (Listof Mat) (Listof Mat)))
(define (make-zero-mat-list mat-list)
  (foldl (lambda ([mat : Mat] [mat-l : (Listof Mat)]) : (Listof Mat)
           (cons (make-matrix
                  (matrix-num-rows mat)
                  (matrix-num-cols mat)
                  0) mat-l)) '() mat-list))

(: dcost (-> neural-network (Listof Mat) (Listof Mat) neural-network))
(define (dcost nn input output)

  (: dcost-rec
     (-> neural-network (Listof Mat) (Listof Mat) (Listof Mat) neural-network
         neural-network))

  (define (dcost-rec nn input output wl bp-gd-acc)
  (match input
    ['() bp-gd-acc]
    [l
    (let* ((fwd-tree (nn-forward (car input) nn))
            (y (car output))
            (res-diff (matrix->list (matrix- (car fwd-tree) y)))
            (bp-gd (dcost-nn fwd-tree wl '() '() res-diff)))

      (begin
         ;; (printf "Samples ~a\n" (car input))
        ;; (printf "Ress diff ~a\n" res-diff)
      (dcost-rec nn (cdr input) (cdr output)
                 wl (nn-apply matrix+ bp-gd bp-gd-acc))))
     ]
    ))

  (let* ((wl (reverse (neural-network-wl nn)))
         (bl (reverse (neural-network-bl nn)))
         (grad-nn (dcost-rec nn input output wl
                             (neural-network
                              (make-zero-mat-list wl)
                              (make-zero-mat-list bl)))))
      
    (nn-map
     (lambda ([m : Mat]) : Mat
       (matrix-scale m (/ 1 (length input)))) grad-nn)))


(: learn (-> neural-network (Listof Mat) (Listof Mat) Real Integer
             neural-network))
(define (learn nn in out rate iter)

  (: learn-rec (-> neural-network (Listof Mat) (Listof Mat) Integer
                   neural-network))
  (define (learn-rec nn in out i)
    (match i
      [0 nn]
      [_
       (let* ((trained-nn (dcost nn in out))
              (rate-nn (nn-map
                        (lambda ([m : Mat]) : Mat
                          (matrix-scale m rate)) trained-nn))
              (new-nn (nn-apply matrix- nn rate-nn)))
         
         (begin
           ;; (printf "W1 ~a \n" (matrix-ref (car (neural-network-wl trained-nn)) 1 0))
           
           (learn-rec new-nn in out (- i 1))))
       ]
      ))

  (learn-rec nn in out iter))

(: perform (-> neural-network (Listof Mat) (Listof Mat) Number))
(define (perform nn in out)
  (match in
    ['() 0]
    [_
     (begin
       (printf "NN's Result: ~a; \nExpected: ~a\n\n"
               (car (nn-forward (car in) nn))
               (car out))
       (perform nn (cdr in) (cdr out)))
     ]
    ))


(: train (-> Integer (Listof Integer) (Listof Mat) (Listof Mat) Real
             neural-network))
(define (train iter-count arch data out train-rate)
  (let*
      (
       (nn (make-nn arch))
       (trained-nn (learn nn data out train-rate iter-count))
       )

    trained-nn
    ))

(define xor-output
  (list
    (matrix [[0 0 0]])
    (matrix [[0 0 1]])
    (matrix [[0 0 1]])
    (matrix [[0 1 0]])
    (matrix [[0 0 1]])
    (matrix [[0 1 0]])
    (matrix [[0 1 0]])
    (matrix [[0 1 1]])
    (matrix [[0 0 1]])
    (matrix [[0 1 0]])
    (matrix [[0 1 0]])
    (matrix [[0 1 1]])
    (matrix [[0 1 0]])
    (matrix [[0 1 1]])
    (matrix [[0 1 1]])
    (matrix [[1 0 0]])
    ))

(define xor-input
  (list 
    (matrix [[0 0 0 0]])
    (matrix [[0 0 0 1]])
    (matrix [[0 0 1 0]])
    (matrix [[0 0 1 1]])
    (matrix [[0 1 0 0]])
    (matrix [[0 1 0 1]])
    (matrix [[0 1 1 0]])
    (matrix [[0 1 1 1]])
    (matrix [[1 0 0 0]])
    (matrix [[1 0 0 1]])
    (matrix [[1 0 1 0]])
    (matrix [[1 0 1 1]])
    (matrix [[1 1 0 0]])
    (matrix [[1 1 0 1]])
    (matrix [[1 1 1 0]])
    (matrix [[1 1 1 1]])
    ))

(define train-in xor-input)
(define train-out xor-output)

(define main
  (let* (
         (arch '(4 16 16 8 3))
         ;; (arch (get-arch (cdr cmd-line)))
         (nn (make-nn arch))
         (trained-nn (learn nn train-in train-out 1 5000))
         )
    (begin
      (perform nn s-in train-out)
      (printf "NN ~a\n" (cost nn train-in train-out))
      (printf "Trained NN ~a\n" (cost trained-nn train-in train-out))
      (perform trained-nn train-in train-out)
      ;; (print-nn (nn-apply matrix- trained-nn nn))
      )))


