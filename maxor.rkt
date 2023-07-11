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

(define train-rate 1e-1)

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

;; (define (cost input 

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

;; (define (learn nn in out)
  ;; )
    
  
(define main
  (let ((nn (make-nn nn-arch)))
    (cost nn xor-input xor-output)))

main
