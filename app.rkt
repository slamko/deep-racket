#!/usr/bin/env racket

#lang racket

(require "deep.rkt")
(require math/matrix)

(define xor-output
  (list
    (matrix [[0 0]])
    (matrix [[0 1]])
    (matrix [[1 0]])
    (matrix [[1 1]])
    ))

(define xor-input
  (list 
    (matrix [[0 0]])
    (matrix [[0 1]])
    (matrix [[1 0]])
    (matrix [[1 1]])))

(define train-in xor-input)
(define train-out xor-output)

(define cmd-line
  (command-line
   #:program "deep.rkt"
   #:args (iter . arch)
   (cons iter arch)))

(define iter-count
  (let ((arg (car cmd-line)))
    (if (string? arg)
        (let ((num (string->number arg)))
          (if (exact-integer? num) num 0 ))
        0
        )))

(define (get-arch cmd-args)
  (foldl
   (lambda (u acc)
     (if (exact-integer? u)
         (cons u acc)
         acc))
   '()
   (map string->number
        (map (lambda (arg)
               (if (string? arg)
                   arg
                   ""))
               cmd-args))))

(define main
  (let* ((arch (get-arch (cdr cmd-line)))
         (nn (make-nn (reverse arch)))
        (trained-nn (learn nn train-in train-out 1 iter-count))
         )
    (begin
      (perform nn train-in train-out)
      (printf "NN ~a\n" (cost nn train-in train-out))
      (printf "Trained NN ~a\n" (cost trained-nn train-in train-out))
      (perform trained-nn train-in train-out)
      ;; (print-nn nn)
      )))

    
