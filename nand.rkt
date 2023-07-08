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

(define or-neuron
  (neuron (random) (random) (random)))

(define nand-neuron
  (neuron (random) (random) (random)))

(define and-neuron
  (neuron (random) (random) (random)))

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

(define (diff data-set n)
  (let* ((yn (neuron-think n (cdr data-set)))
        (y (car data-set))
        (d (- y yn)))

    (* d d)))

(define (cost data-list n)
  (define (cost-rec data err n)
    (match data
      ['() err]
      [l
       (cost-rec (cdr l) (+ err (diff (car l) n)) n)
       ]
      ))

  (cost-rec data-list 0 n))

(define (train data n rate)
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

        (start-cost (cost data n))
        (cost-w1+ (cost data neuron-w1+))
        (cost-w2+ (cost data neuron-w2+))
        (cost-b+ (cost data neuron-b+))

        (dw1 (/ (- cost-w1+ start-cost) dstep))
        (dw2 (/ (- cost-w2+ start-cost) dstep))
        (db  (/ (- cost-b+ start-cost) dstep)))

    (neuron (- (neuron-w1 n) (* dw1 rate))
            (- (neuron-w2 n) (* dw2 rate))
            (- (neuron-b n) (* db rate)))
  ))

(define (perform data n)
  (match data
    ['() ""]
    [l
     (let ((y (car (car l)))
           (yn (neuron-think n (cdr (car l)))))
       (printf "real y = ~a . neuron's y = ~a\n" y yn))
     (perform (cdr l) n)
     ]
    ))
   
(define (learn data)
  (define (learn-rec data err n iter)
    (match iter
      [j
       #:when (or (= j 0) (< err 1e-4))
       n]
      [i
       (let ((trained-n (train data n train-rate)))
         ;; (printf "~a \n" (cost data trained-n))
         (learn-rec data (cost data trained-n) trained-n (- i 1)))
         ]
      ))

  (learn-rec data 1 or-neuron (* 1000 1000)))

(define main
  (begin
    (perform train-data or-neuron)
    (printf "-----------------------------------------\n")
    (let ((n (learn train-data)))
      (perform train-data n)
      (printf "~a \n" (cost train-data n)))))

main
