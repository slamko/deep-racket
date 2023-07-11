#!/usr/bin/env racket

#lang typed/racket
;; #lang racket

(require math)
(require math/matrix)

(define-type Mat (Matrix Real))

(: xor-output (Listof Real))
(define xor-output
  '(0 1 1 0))

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

;; (define train-data xor-train-data)

(define input-data xor-input)

(define out-data xor-output)

(define train-rate 1e-1)

(define dstep 1e-1)

(define-type Data (Listof (Listof Real)))

(: sigmoid (-> Real Real))
(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(: forward-layer (-> Mat Mat Mat Mat))
(define (forward-layer in w b)
  (matrix-map sigmoid (matrix+ (matrix* in w) b)))

(: forward (-> Mat (Listof Mat) (Listof Mat) Real))
(define (forward in wl bl)
  (match wl
    ['() (car (matrix->list in))]
    [l
     (let ((a (forward-layer in (car wl) (car bl))))
       (forward a (cdr wl) (cdr bl)))
     ]
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
        (cons (make-matrix (car (cdr n-arch)) (car n-arch) (random)) wl))
       ]
      ))
  
  (: make-bl-rec (-> (Listof Integer) (Listof Mat) (Listof Mat)))
  (define (make-bl-rec n-arch bl)
    (match n-arch
      [(list a) bl]
      [n
       (make-bl-rec
        (cdr n-arch)
        (cons (make-matrix 1 (car n-arch) (random)) bl))
       ]
      ))
  
  (let ((rev-arch (reverse arch)))
    (neural-network (make-wl-rec rev-arch '()) (make-bl-rec rev-arch '()))))

(: cost (-> neural-network (Listof Mat) (Listof Real) Real))
(define (cost nn input output)

  (: cost-rec (-> neural-network (Listof Mat) (Listof Real) Real Real))
  (define (cost-rec nn input output err)
  (match input
    ['() err]
    [l
     (let* ((nn-y (forward (car input)
                             (neural-network-wl nn)
                             (neural-network-bl nn)))
           (y (car output))
           (diff (- y nn-y)))
       (cost-rec nn (cdr input) (cdr output) (+ err (* diff diff))))
     ]
    ))

  (cost-rec nn input output 0))

(: list-back->mat (-> Mat (Listof Real) Mat))
(define (list-back->mat mat ls) 
  (list->matrix (matrix-num-rows mat) (matrix-num-cols mat) ls))

(: mat-size (-> Mat Integer))
(define (mat-size m)
  (* (matrix-num-cols m) (matrix-num-rows m)))

(: mat-step (-> (Listof Mat) Real (Listof (Listof Mat))))
(define (mat-step mat-list dstep)

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
              (let* ((new-lr
                      (list-set ml i (+ (list-ref ml i) dstep)))
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

;; (: mstep (-> Mat (Listof Mat)))
;; (define (mstep m)
  ;; (mat-step m dstep))

;; (: finite-diff (-> (Listof 
;; (define (finite-diff wl bl)
  ;; (map mstep wl))

(: learn (-> neural-network (Listof Mat) (Listof Real) (Listof (Listof Mat))))
(define (learn nn in out)
  (mat-step (neural-network-wl nn) 1)
  )
    
  
(define main
  (let ((nn (make-nn nn-arch)))
    (learn nn input-data out-data)))

main
