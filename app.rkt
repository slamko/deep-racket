#!/usr/bin/env racket

#lang racket

(require "deep.rkt")
(require math/matrix)
(require csv-reading)
(require json)

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

(define (wr-to-json expr fname)
  (let ((json-out (open-output-file
                   fname
                   #:mode 'text
                   #:exists 'append)))
    (write-json expr json-out)))

(define (nn->jsexpr nn)
  (hash
   'weights (map (lambda (m) (matrix->list m)) (neural-network-wl nn))
   'biases  (map (lambda (m) (matrix->list m)) (neural-network-bl nn))
   ))

(define (save-to-json nn fname)
  (let ((json-out (open-output-file
                   fname
                   #:mode 'text
                   #:exists 'truncate)))

    (begin 
      (fprintf json-out (jsexpr->string (nn->jsexpr nn)))
      (close-output-port json-out))
    ))


(define (restore-from-json nn fname)
  (let* ((json-in (open-input-file fname))
        (json-str (file->string fname))
        (nn-hash (string->jsexpr json-str)))

    ;; (for (((key val) (in-hash nn-hash)))
      ;; (printf "~a = ~a~%" key val))

    (neural-network
     (map (lambda (l mat)
            (list->matrix
             (matrix-num-rows mat)
             (matrix-num-cols mat)
             l))

          (hash-ref nn-hash 'weights)
          (neural-network-wl nn))

     (map (lambda (l mat)
            (list->matrix
             (matrix-num-rows mat)
             (matrix-num-cols mat)
             l))

          (hash-ref nn-hash 'biases)
          (neural-network-bl nn)))
    ))

(struct person (first-name last-name age country))
(define (person->jsexpr p)
  (hasheq 'first-name (person-first-name p)
          'last-name (person-last-name p)
          'age (person-age p)
          'country (person-country p)))
(define cky (person "Chris" "Jester-Young" 33 "New Zealand"))

(define main
  (let* (
         (arch '(784 16 16 10))
         ;; (arch (get-arch (cddr cmd-line)))
         (nn (restore-from-json (make-nn arch) "save.json"))
         (train-data (parse-csv-train-data train-data-file-name)) 
         ;; (train-data xor-data)
         (trained-nn (learn nn train-data 1 3))
         )
    (begin
      ;; (perform trained-nn train-data)
      (printf "NN ~a\n" (cost nn train-data))
      (printf "Trained NN ~a\n" (cost trained-nn train-data))

      (save-to-json trained-nn "save.json")
      )))



    
