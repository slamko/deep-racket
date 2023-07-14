#!/usr/bin/env racket

#lang racket

(require "deep.rkt")
(require math/matrix)

(define xor-output
  (list
    (matrix [[0]])
    (matrix [[1]])
    (matrix [[1]])
    (matrix [[0]])))

(define xor-input
  (list 
    (matrix [[0 0]])
    (matrix [[0 1]])
    (matrix [[1 0]])
    (matrix [[1 1]])))

(define nn (make-nn '(2 5 1)))
(define trained-nn (learn nn xor-input xor-output 1 (* 10 1000)))

(perform trained-nn xor-input xor-output)

