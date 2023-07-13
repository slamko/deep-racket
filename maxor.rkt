#!/usr/bin/env racket

#lang typed/racket
;; #lang racket

(require math)
(require math/matrix)

(define-type Mat (Matrix Real))
(define-type Weight Real)
(define-type Bias Real)

(: xor-output (Listof Mat))
(define xor-output
  (list
    (matrix [[0]])
    (matrix [[1]])
    (matrix [[1]])
    (matrix [[0]])))

(: xor-input (Listof Mat))
(define xor-input
  (list 
    (matrix [[0 0]])
    (matrix [[0 1]])
    (matrix [[1 0]])
    (matrix [[1 1]])))

(: nn-arch (Listof Integer))
(define nn-arch '(2 2 1))

(struct neural-network
  (
   [wl : (Listof Mat)]
   [bl : (Listof Mat)]))

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

;; (define train-data xor-train-data)

(define input-data xor-input)

(define out-data xor-output)

(define train-rate 1e-1)

(: sigmoid (-> Real Real))
(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

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

(: randf (-> () Real))
(define (randf)
  (- (* (random) 2) 1))

(: make-nn (-> (Listof Integer) neural-network))
(define (make-nn arch)
  (: make-wl-rec (-> (Listof Integer) (Listof Mat) (Listof Mat)))
  (define (make-wl-rec n-arch wl)
    (match n-arch
      [(list a) wl]
      [n
       (make-wl-rec
        (cdr n-arch)
        (cons (make-matrix (car (cdr n-arch)) (car n-arch) (randf)) wl))
       ]
      ))
  
  (: make-bl-rec (-> (Listof Integer) (Listof Mat) (Listof Mat)))
  (define (make-bl-rec n-arch bl)
    (match n-arch
      [(list a) bl]
      [n
       (make-bl-rec
        (cdr n-arch)
        (cons (make-matrix 1 (car n-arch) (randf)) bl))
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

(: cost (-> neural-network (Listof Mat) (Listof Mat) Real))
(define (cost nn input output)

  (: cost-rec (-> neural-network (Listof Mat) (Listof Mat) Real Real))
  (define (cost-rec nn input output err)
  (match input
    ['() err]
    [l
     (let* ((nn-y (car (forward (car input)
                             (neural-network-wl nn)
                             (neural-network-bl nn))))
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
           (pd-ai-prev (* 2 diff-i ai (- 1 ai)
                          (list-ref w-acc-l i)))
           (wlist-upd (list-set w-acc-l i wi-pd))
           )
       
       (begin
         ;; (printf "~a\n" wlist-upd)
       (dcost-neuron
        wlist-upd
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
            (cur-row-l (matrix->list (matrix-col w-mat (- i 1))))
            (neuron-bp
             (dcost-neuron cur-row-l diff ai ai-prev-l
                           next-diff-l
                           (- w-row 1)))
            (neuron-dcost (backprop-neuron-w-list neuron-bp))
            (neuron-diff-l (backprop-neuron-pd-prev neuron-bp))
            (n-dcost-mat (list->matrix w-row 1 neuron-dcost))
            (bias-gd (* 2 diff ai (- 1 ai))))
       
       (begin
         ;; (printf "Current row: ~a\n" (- w-row 1))
         ;; (printf "Current col: ~a\n" i)
         ;; (printf "Current activation: ~a\n" ai-l)
         (dcost-layer w-row (- i 1) w-col
                      (matrix-set-col w-mat (- i 1) n-dcost-mat)
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
             (dcost-layer w-rows w-cols w-cols cur-wmat
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
         ;; (printf "Cur wmat~a\n" cur-wmat)
         (dcost-nn (cdr forward-list) (cdr wmat-list)
                   (cons bp-wgrad-mat wgrad-mat-list-acc)
                   (cons bp-bgrad-mat bgrad-mat-list-acc)
                   bp-next-diff-l)))
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
    (let* ((fwd-tree (forward (car input)
                             (neural-network-wl nn)
                             (neural-network-bl nn)))
            (y (car output))
            (res-diff (matrix->list (matrix- (car fwd-tree) y)))
            (bp-gd (dcost-nn fwd-tree wl '() '() res-diff)))

      (dcost-rec nn (cdr input) (cdr output)
                 wl
                 (neural-network
                  (map matrix+
                       (neural-network-wl bp-gd)
                       (neural-network-wl bp-gd-acc))
                  (map matrix+
                       (neural-network-bl bp-gd)
                       (neural-network-bl bp-gd-acc)))))

     ]
    ))

  (let* ((wl (reverse (neural-network-wl nn)))
         (bl (reverse (neural-network-bl nn)))
         (grad-nn (dcost-rec nn input output wl
                             (neural-network
                              (make-zero-mat-list wl)
                              (make-zero-mat-list bl)))))
      
      (neural-network
       (map
        (lambda ([m : Mat]) : Mat
          (matrix-scale m (/ 1 (length input))))
        (neural-network-wl grad-nn))
       (map
        (lambda ([m : Mat]) : Mat
          (matrix-scale m (/ 1 (length input))))
        (neural-network-bl grad-nn)))))
#|
(: mat-step (-> (Listof Mat) Real (Listof (Listof Mat))))
(define (mat-step base-nn mat-list dstep)

  (: step-rec
     (-> (Listof Mat) (Listof (Listof Mat)) Integer (Listof (Listof Mat))))
  (define (step-rec lm ml-acc mat-iter)
    (match mat-iter
      [-1 ml-acc]
      [_
       (let* ((cur-m (list-ref lm mat-iter))
              (lr (matrix->list cur-m)))
         
         (: step-one-mat
            (-> (Listof Real) (Listof (Listof Mat)) Integer
                (Listof (Listof Mat))))
         (define (step-one-mat ml acc iter)
           (match iter
             [-1 acc]
             [i
              (let* (
                     (new-lr
                      (list-set ml i (- (list-ref ml i) )))
                     (new-ml
                      (list-set mat-list mat-iter
                                (list-back->mat cur-m new-lr))))

                (step-one-mat ml (cons new-ml acc) (- i 1)))
              ]
             ))
         
           (step-rec
            lm
            (append (step-one-mat lr '() (- (mat-size cur-m) 1)) ml-acc)
            (- mat-iter 1)))
       ]
      ))

  (step-rec mat-list '() (- (length mat-list) 1)))
|#

;; (: mstep (-> Mat (Listof Mat)))
;; (define (mstep m)
  ;; (mat-step m dstep))

;; (: finite-diff (-> (Listof 
;; (define (finite-diff wl bl)
  ;; (map mstep wl))
#|
(: diff (-> neural-network (Listof Mat) (Listof Real) (Listof (Listof Mat))))
(define (diff nn in out)
  (let ((wll (mat-step (neural-network-wl nn) dstep))
        (bll (mat-step (neural-network-bl nn) dstep)))

    (define (diff-rec wll bll nn-acc in out)
      (match wll
        ['() nn-acc]
        [_
         (let ((base-cost (cost nn in out))
              (cost+
               (cost
                (neural-network
                 (car wll)
                 (neural-network-bl nn))
                in out)))
           (diff-rec (cdr wll) bll ()
               
    )
  |#


(: learn (-> neural-network (Listof Mat) (Listof Mat) neural-network))
(define (learn nn in out)

  (: learn-rec (-> neural-network (Listof Mat) (Listof Mat) Integer
                   neural-network))
  (define (learn-rec nn in out i)
    (match i
      [0 nn]
      [_
       (let* ((trained-nn (dcost nn in out))
              (new-nn
               (neural-network
                (map matrix- (neural-network-wl nn)
                     (map (lambda ([m : Mat]) : Mat
                            (matrix-scale m train-rate))
                          (neural-network-wl trained-nn)))
                
                (map matrix- (neural-network-bl nn)
                     (map (lambda ([m : Mat]) : Mat
                            (matrix-scale m train-rate))
                          (neural-network-bl trained-nn))))))
         
         (learn-rec new-nn in out (- i 1)))
       ]
      ))

  (learn-rec nn in out (* 10 100)))
              
(define main
  (let* ((nn (make-nn nn-arch))
         (trained-nn (learn nn input-data out-data))
         (grad (dcost nn input-data out-data))
         (fwd-tree
          (forward (car input-data)
                   (neural-network-wl nn)
                   (neural-network-bl nn)))
         )
    (begin
      (printf "~a\n" (cost nn input-data out-data))
      (printf "~a\n" (cost trained-nn input-data out-data))
      ;; (forward (car input-data) (neural-network-wl nn) (neural-network-bl nn))
      ;; (neural-network-wl grad)
      ;; (reverse (neural-network-wl nn))

      ;; (printf "~a\n" (car out-data))
      ;; (printf "~a\n" (car fwd-tree))
      ;; (matrix->list (matrix- (car fwd-tree) (car out-data)))
      ;; (map (lambda ([m : Mat]) : Mat
             ;; (matrix-scale m )
             ;; )
         ;; (neural-network-wl grad))
      ;; (neural-network-bl grad)
      ;; (neural-network-wl nn)
      )))

main
