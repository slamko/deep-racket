#!/usr/bin/env racket

#lang racket

(require "deep.rkt")
(require math/matrix)
(require csv-reading)

(define xor-data
  (list 
    (cons (matrix [[0]]) (matrix [[0 0]]))
    (cons (matrix [[1]]) (matrix [[0 1]]))
    (cons (matrix [[1]]) (matrix [[1 0]]))
    (cons (matrix [[0]]) (matrix [[1 1]]))
    ))

(define adder-output
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

(define adder-input
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

(define cmd-line
  (command-line
   #:program "deep.rkt"
   #:args (train-data-file iter . arch)
   (cons train-data-file (cons iter arch))))

(define iter-count
  (let ((arg (cadr cmd-line)))
    (if (string? arg)
        (let ((num (string->number arg)))
          (if (exact-integer? num) num 0 ))
        0
        )))

(define train-data-file-name
  (let ((arg (car cmd-line)))
    (if (string? arg) arg "")))

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

(define (parse-csv-train-data fname)
  (csv-map
   (lambda (csv-row) 
     (let ((real-list
            (map
             (lambda (val) 
               (let ((num (string->number val)))
                 (if (real? num) num 0)))
             csv-row)))
       (begin 
       (cons
        (list->matrix
         1 10
         (list-set (matrix->list (make-matrix 1 10 0)) (car real-list) 1))
        (list->matrix 1 784 (cdr real-list))))))
   (make-csv-reader (open-input-file fname))))


(define main
  (let* (
         ;; (arch '(784 16 16 10))
         (arch '(2 3 1))
         ;; (arch (get-arch (cddr cmd-line)))
         (nn (make-nn arch))
         ;; (train-data (parse-csv-train-data train-data-file-name)) 
         (trained-nn (learn nn xor-data 1 1000))
         )
    (begin
      (printf "NN ~a\n" (cost nn xor-data))
      (printf "Trained NN ~a\n" (cost trained-nn xor-data))
      ;; (perform trained-nn train-data)
      ;; (print (csv->list (make-csv-reader (open-input-file train-data-file-name))))
      ;; (print-nn (nn-apply matrix- trained-nn nn))
      )))



    
