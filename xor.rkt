#!/usr/bin/env racket

#lang typed/racket
;; #lang racket

(require math)
(require math/matrix)

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

(struct neuron ([w1 : Real] [w2 : Real] [b : Real]))

(struct nxor ([a : neuron] [b : neuron] [y : neuron]))

(define or-neuron
  (neuron (random) (random) (random)))

(define nand-neuron
  (neuron (random) (random) (random)))

(define and-neuron
  (neuron (random) (random) (random)))

(define neural-network (nxor or-neuron nand-neuron and-neuron))

(define train-rate 1e-1)


(define-type Data (Listof (Listof Real)))

(: sigmoid (-> Real Real))
(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(: neuron-think (-> neuron (Listof Real) Real))
(define (neuron-think n input)
  (sigmoid
   (+
    (* (car input) (neuron-w1 n))
    (* (car (cdr input)) (neuron-w2 n))
    (neuron-b n)
    )))

(: xor-think (-> nxor (Listof Real) Real))
(define (xor-think xor-nn input)
  (neuron-think
   (nxor-y xor-nn)
   (list
    (neuron-think
     (nxor-a xor-nn) input)
    (neuron-think 
     (nxor-b xor-nn) input))))  

(: xor-diff (-> (Listof Real) nxor Real))
(define (xor-diff data-set nn)
  (let* ((yn (xor-think nn (cdr data-set)))
        (y (car data-set))
        (d (- y yn)))

    (* d d)))

(: cost (-> Data nxor Real))
(define (cost data-list network)

  (: cost-rec (-> Data Real nxor Real))
  (define (cost-rec data err network)
    (match data
      ['() err]
      [l
       (cost-rec (cdr l) (+ err (xor-diff (car l) network)) network)
       ]
      ))

  (cost-rec data-list 0 network))

(: train-neuron
   (-> Data nxor neuron (-> neuron nxor) Real neuron))

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

(: train-net (-> Data nxor Real nxor))
(define (train-net data network rate)
  (nxor
   (train-neuron data network
                 (nxor-a network)
                 (lambda ([neuron : neuron]) : nxor
                   (struct-copy nxor network
                                [a neuron]))
                 rate)

   (train-neuron data network
                 (nxor-b network)
                 (lambda ([neuron : neuron]) : nxor
                   (struct-copy nxor network
                                [b neuron]))
                 rate)

   (train-neuron data network
                 (nxor-y network)
                 (lambda ([n : neuron]) : nxor
                   (struct-copy nxor network
                                [y n]))
                 rate)
   ))

(: perform (-> Data nxor String))
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
   

(: learn (-> Data nxor))
(define (learn data)

  (: learn-rec (-> Data Real nxor Number nxor))
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
    (printf "Before training: \n")
    (perform train-data neural-network)
    (printf "-----------------------------------------\n")
    (let ((nn (learn train-data)))
      ;; (printf "w1 ~a w2 ~a" (neuron-w1 (nxor-a nn))
              ;; (neuron-w2 (nxor-a nn)))
      (printf "After training: \n")
      (perform train-data nn)
      (printf "\nModel error = ~a \n" (cost train-data nn)))))

main
