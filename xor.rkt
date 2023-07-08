#!/usr/bin/env racket

#lang racket

(require math)

(define or-train-data
  (list
   '(0 0 0)
   '(1 0 1)
   '(1 1 0)
   '(1 1 1)))

(define nand-train-data
  (list
   '(1 0 0)
   '(1 0 1)
   '(1 1 0)
   '(0 1 1)))

(define and-train-data
  (list
   '(0 0 0)
   '(0 0 1)
   '(0 1 0)
   '(1 1 1)))

(define xor-train-data
  (list
   '(0 0 0)
   '(1 0 1)
   '(1 1 0)
   '(0 1 1)))

(define train-data xor-train-data)

(struct neuron (w1 w2 b))

(struct nxor (a b y))

(define or-neuron
  (neuron (random) (random) (random)))

(define nand-neuron
  (neuron (random) (random) (random)))

(define and-neuron
  (neuron (random) (random) (random)))

(define neural-network (nxor or-neuron nand-neuron and-neuron))

(define train-rate 1e-1)

(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(define (neuron-think n input)
  (sigmoid
   (+
    (* (car input) (neuron-w1 n))
    (* (car (cdr input)) (neuron-w2 n))
    (neuron-b n)
    )))

(define (xor-think xor-nn input)
  (neuron-think
   (nxor-y xor-nn)
   (list
    (neuron-think
     (nxor-a xor-nn) input)
    (neuron-think 
     (nxor-b xor-nn) input))))  

(define (xor-diff data-set nn)
  (let* ((yn (xor-think nn (cdr data-set)))
        (y (car data-set))
        (d (- y yn)))

    (* d d)))

(define (cost data-list network)
  (define (cost-rec data err network)
    (match data
      ['() err]
      [l
       (cost-rec (cdr l) (+ err (xor-diff (car l) network)) network)
       ]
      ))

  (cost-rec data-list 0 network))

(define (train-neuron data network n insert-neuron-f rate)
  (let* (
        (dstep 1e-3)
        (neuron-w1+ (neuron
                     (+ (neuron-w1 n) dstep)
                     (neuron-w2 n)
                     (neuron-b n)))

        (neuron-w2+ (neuron
                     (neuron-w1 n)
                     (+ (neuron-w2 n) dstep)
                     (neuron-b n)))

        (neuron-b+ (neuron
                     (neuron-w1 n)
                     (neuron-w2 n)
                     (+ (neuron-b n) dstep)))

        (start-cost (cost data network))
        (cost-w1+ (cost data (insert-neuron-f neuron-w1+)))
        (cost-w2+ (cost data (insert-neuron-f neuron-w2+)))
        (cost-b+ (cost data (insert-neuron-f neuron-b+)))

        (dw1 (/ (- cost-w1+ start-cost) dstep))
        (dw2 (/ (- cost-w2+ start-cost) dstep))
        (db  (/ (- cost-b+ start-cost) dstep)))

    (neuron (- (neuron-w1 n) (* dw1 rate))
            (- (neuron-w2 n) (* dw2 rate))
            (- (neuron-b n) (* db rate)))
  ))

(define (train-net data network rate)
  (nxor
   (train-neuron data network
                 (nxor-a network)
                 (lambda (neuron)
                   (struct-copy nxor network
                                [a neuron]))
                 rate)

   (train-neuron data network
                 (nxor-b network)
                 (lambda (neuron)
                   (struct-copy nxor network
                                [b neuron]))
                 rate)

   (train-neuron data network
                 (nxor-y network)
                 (lambda (neuron)
                   (struct-copy nxor network
                                [y neuron]))
                 rate)
   ))

(define (perform data nn)
  (match data
    ['() ""]
    [l
     (let ((y (car (car l)))
           (yn (xor-think nn (cdr (car l)))))
       (printf "real y = ~a . neuron's y = ~a\n" y yn))
     (perform (cdr l) nn)
     ]
    ))
   
(define (learn data)
  (define (learn-rec data err nn iter)
    (match iter
      [j
       #:when (or (= j 0) (< err 1e-4))
       nn]
      [i
       (let (
             (trained-nn (train-net data nn train-rate)))
         ;; (printf "~a \n" (cost data trained-n))
         (learn-rec data (cost data trained-nn) trained-nn (- i 1)))
         ]
      ))

  (learn-rec data 1 neural-network (* 1000 1000)))

(define main
  (begin
    (perform train-data neural-network)
    (printf "-----------------------------------------\n")
    (let ((nn (learn train-data)))
      (perform train-data nn)
      (printf "~a \n" (cost train-data nn)))))

main
