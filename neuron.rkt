#!/bin/env racket

#lang racket

(define train-data
  (list
   '(2 1)
   '(10 5)
   '(22 11)
   ;; '(46 23)
   '(48 24)
   ;; '(662 331)
   ;; '(808 404)
   ;; '(12 6)
   ;; '(14 7)
   ;; '(16 8)
   ;; '(36 18)
   ))

(define rate 0.001)
(define weight (random))
(define bias (random))

(define (think x1 w1 b)
  (+ (* x1 w1) b))

(define (cadar l)
  (car (cdr (car l))))

(define (cost data w b)
  (define (cost-rec data w b err)
    (match data
        ['() err]
        [l
         (let* (
                (y (car (car l)))
                (y1 (think (cadar l) w b))
                (dw (- y y1)))

           (cost-rec (cdr l) w b (+ err (* dw dw))))
         ]
        ))
  
  (cost-rec data w b 0))

(define (train data w b)
  (define (train-rec data cur-err w b iter)
    (match cur-err
      [train-err
       #:when (or (= iter 0) (< train-err 1e-5))
       (list b w)]
      [_
        (let* (
            (dstep 1e-3)
            (w-err-plus (cost data (+ w dstep) b)) 
            (err (cost data w b))
            (b-err-plus (cost data w (+ b dstep))) 

            (w-deriv (/ (- w-err-plus err) dstep))
            (b-deriv (/ (- b-err-plus err) dstep))
            )

        ;; (printf "error: ~a\n" err)
        (train-rec data err (- w (* w-deriv rate)) (- b (* b-deriv rate)) (- iter 1)))
      ]
    ))

  (train-rec data 10 w b (* 10 10)))
       
  
(define (learn data)
  (train data weight bias))

(define main
  (let* (
        (input 192)
        (trained (learn train-data))
        (trained-weight (car (cdr trained)))
        (trained-bias (car trained)))

    (printf "w = ~a ; b = ~a\n" trained-weight trained-bias);
    (printf "~a * 2 = ~a\n" input (think input trained-weight trained-bias))))

main
