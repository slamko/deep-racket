#!/usr/bin/env racket

#lang typed/racket
;; #lang racket

(require math)
(require math/matrix)
;; (require racket/struct)
;; (require trace)

(define-type Mat (Matrix Real))
(define-type Weight Real)
(define-type Bias Real)

(: xor-output (Listof Mat))
(define xor-output
  (list
    (matrix [[0]])
    (matrix [[1]])
    (matrix [[1]])
    (matrix [[0]])))

(: xor-input (Listof Mat))
(define xor-input
  (list 
    (matrix [[0 0]])
    (matrix [[0 1]])
    (matrix [[1 0]])
    (matrix [[1 1]])))

(: nn-arch (Listof Integer))
(define nn-arch '(2 5 1))

(struct neural-network
  (
   [wl : (Listof Mat)]
   [bl : (Listof Mat)]))

(: print-nn (-> neural-network Void))
(define (print-nn nn)
   (printf "Weights list: \n~a \nBiases list: \n~a\n"
              (neural-network-wl nn)
              (neural-network-bl nn)))

(struct backprop-nn
  (
   [wmat-list : (Listof (Matrix Weight))]
   [bmat-list : (Listof (Matrix Bias))]
   ))

(struct backprop-layer
  (
   [pd-prev : (Listof Real)]
   [w-mat : (Matrix Weight)]
   [b-mat : (Matrix Bias)]
   ))

(struct backprop-neuron
  (
   [w-list : (Listof Weight)]
   [pd-prev : (Listof Real)]
   ))

;; (define train-data xor-train-data)

(define input-data xor-input)

(define out-data xor-output)

(define train-rate 1)

(: sigmoid (-> Real Real))
(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(: forward-layer (-> Mat Mat Mat Mat))
(define (forward-layer in w b)
  (matrix-map sigmoid (matrix+ (matrix* in w) b)))

(: forward (-> Mat (Listof Mat) (Listof Mat) (Listof Mat)))
(define (forward in wl bl)

  (: forward-acc (-> Mat (Listof Mat) (Listof Mat) (Listof Mat) (Listof Mat)))
  (define (forward-acc in wl bl a-acc)
    (match wl
        ['() a-acc]
        [l
         (let ((a (forward-layer in (car wl) (car bl))))
           (forward-acc a (cdr wl) (cdr bl) (cons a a-acc)))
        ]
        ))

  (forward-acc in wl bl (list in)))


(: nn-forward (-> Mat neural-network (Listof Mat)))
(define (nn-forward in nn)
  (forward in
           (neural-network-wl nn)
           (neural-network-bl nn)))

(: randf (-> Real Real))
(define (randf r)
  (- (* (random) 2) 1))

(: make-random-mat (-> Integer Integer Mat))
(define (make-random-mat row col)
  (matrix-map randf (make-matrix row col 0) ))

(: make-nn (-> (Listof Integer) neural-network))
(define (make-nn arch)
  (: make-wl-rec (-> (Listof Integer) (Listof Mat) (Listof Mat)))
  (define (make-wl-rec n-arch wl)
    (match n-arch
      [(list a) wl]
      [n
       (make-wl-rec
        (cdr n-arch)
        (cons (make-random-mat (car (cdr n-arch)) (car n-arch)) wl))
       ]
      ))
  
  (: make-bl-rec (-> (Listof Integer) (Listof Mat) (Listof Mat)))
  (define (make-bl-rec n-arch bl)
    (match n-arch
      [(list a) bl]
      [n
       (make-bl-rec
        (cdr n-arch)
        (cons (make-random-mat 1 (car n-arch)) bl))
       ]
      ))
  
  (neural-network (make-wl-rec arch '()) (make-bl-rec arch '())))

(: list-back->mat (-> Mat (Listof Real) Mat))
(define (list-back->mat mat ls) 
  (list->matrix (matrix-num-rows mat) (matrix-num-cols mat) ls))

(: mat-size (-> Mat Integer))
(define (mat-size m)
  (* (matrix-num-cols m) (matrix-num-rows m)))

(: nn-map (-> (-> Mat Mat)
              neural-network neural-network))
(define (nn-map proc nn)
    (neural-network
     (map proc (neural-network-wl nn))
     (map proc (neural-network-bl nn))))

(: cost (-> neural-network (Listof Mat) (Listof Mat) Real))
(define (cost nn input output)

  (: cost-rec (-> neural-network (Listof Mat) (Listof Mat) Real Real))
  (define (cost-rec nn input output err)
  (match input
    ['() err]
    [l
     (let* ((nn-y (car (nn-forward (car input) nn)))
           (y (car output))
           (diff (apply + (matrix->list (matrix- nn-y y)))))
       (cost-rec nn (cdr input) (cdr output) (+ err (* diff diff))))
     ]
    ))

  (/ (cost-rec nn input output 0) (length input)))

(: dcost-neuron
 (-> (Listof Real) Real Real (Listof Real) (Listof Real) Integer
     backprop-neuron))
(define (dcost-neuron w-acc-l diff-i ai ai-prev-l
                      pd-prev-list-acc i)
  (match i
    [-1 (backprop-neuron w-acc-l pd-prev-list-acc)]
    [_
     (let* ((ai-prev (list-ref ai-prev-l i))
           (wi-pd (* 2 diff-i ai (- 1 ai) ai-prev))
           (last-pd-ai-prev (list-ref pd-prev-list-acc i))
           (cur-w (list-ref w-acc-l i))
           (pd-ai-prev (* 2 diff-i ai (- 1 ai) cur-w))
           )
       
       (begin
         ;; (printf "Diff ~a, ai ~a, ai-prev ~a wi ~a\n" diff-i ai ai-prev wi-pd)
       (dcost-neuron
        (list-set w-acc-l i wi-pd)
        diff-i
        ai
        ai-prev-l
        (list-set pd-prev-list-acc i
                  (+ pd-ai-prev last-pd-ai-prev))
        (- i 1))))
     ]
    ))



;; (define (row-list->mat row-list)
  ;; (foldl 

(: dcost-layer
   (-> Integer Integer Integer (Matrix Weight) (Listof Real) (Listof Real) (Listof Real)
       (Listof Real) (Listof Real) backprop-layer))
(define (dcost-layer w-row i w-col w-mat b-acc-list next-diff-l ai-l
                     diff-l ai-prev-l)
  (match ai-l
    ['() (backprop-layer
          next-diff-l
          w-mat
          (list->matrix 1 w-col b-acc-list)
          )]
    [_
     
     (let* ((ai (car ai-l))
            (diff (car diff-l))
            (cur-row-l (matrix->list (matrix-col w-mat (- i 1))))
            (neuron-bp
             (dcost-neuron cur-row-l diff ai ai-prev-l
                           next-diff-l
                           (- w-row 1)))
            (neuron-dcost (backprop-neuron-w-list neuron-bp))
            (neuron-diff-l (backprop-neuron-pd-prev neuron-bp))
            (n-dcost-mat (list->matrix w-row 1 neuron-dcost))
            (bias-gd (* 2 diff ai (- 1 ai))))
       
       (begin
         ;; (printf "Current next: ~a\n" next-diff-l)
         ;; (printf "Current activation: ~a\n" neuron-diff-l)
         (dcost-layer w-row (- i 1) w-col
                      (matrix-set-col w-mat (- i 1) n-dcost-mat)
                      (cons bias-gd b-acc-list)
                      neuron-diff-l
                      (cdr ai-l) (cdr diff-l) ai-prev-l)))
       ]
    ))

(: dcost-nn
   (-> (Listof Mat) (Listof Mat) (Listof Mat) (Listof Mat) (Listof Real) neural-network))
(define (dcost-nn forward-list wmat-list
                  wgrad-mat-list-acc bgrad-mat-list-acc diff-l)
  (match forward-list
    [(list a)
     (neural-network
      wgrad-mat-list-acc
      bgrad-mat-list-acc)
      ]
    [l
     (let* (
            (cur-wmat (car wmat-list))
            (w-rows (matrix-num-rows cur-wmat))
            (w-cols (matrix-num-cols cur-wmat))
            (layer-bp
             (dcost-layer w-rows w-cols w-cols cur-wmat
                          '()
                          (make-list w-rows 0)
                          (matrix->list (car forward-list))
                          diff-l
                          (matrix->list (cadr forward-list))))
            (bp-next-diff-l (backprop-layer-pd-prev layer-bp))
            (bp-wgrad-mat (backprop-layer-w-mat layer-bp))
            (bp-bgrad-mat (backprop-layer-b-mat layer-bp))
            )
       
       (begin
         ;; (printf "Cur diff~a\n" diff-l)
         ;; (printf "Cur fwd list~a\n" (car forward-list))
         (dcost-nn (cdr forward-list) (cdr wmat-list)
                   (cons bp-wgrad-mat wgrad-mat-list-acc)
                   (cons bp-bgrad-mat bgrad-mat-list-acc)
                   (map (lambda ([d : Real]) : Real (* d 1)) bp-next-diff-l))))
     ]
    ))

(: make-zero-mat-list (-> (Listof Mat) (Listof Mat)))
(define (make-zero-mat-list mat-list)
  (foldl (lambda ([mat : Mat] [mat-l : (Listof Mat)]) : (Listof Mat)
           (cons (make-matrix
                  (matrix-num-rows mat)
                  (matrix-num-cols mat)
                  0) mat-l)) '() mat-list))

(: dcost (-> neural-network (Listof Mat) (Listof Mat) neural-network))
(define (dcost nn input output)

  (: dcost-rec
     (-> neural-network (Listof Mat) (Listof Mat) (Listof Mat) neural-network
         neural-network))

  (define (dcost-rec nn input output wl bp-gd-acc)
  (match input
    ['() bp-gd-acc]
    [l
    (let* ((fwd-tree (nn-forward (car input) nn))
            (y (car output))
            (res-diff (matrix->list (matrix- (car fwd-tree) y)))
            (bp-gd (dcost-nn fwd-tree wl '() '() res-diff)))

      (begin
        ;; (printf "Ress diff ~a\n" res-diff)
      (dcost-rec nn (cdr input) (cdr output)
                 wl (neural-network
                     (map matrix+
                          (neural-network-wl bp-gd)
                          (neural-network-wl bp-gd-acc))
                     (map matrix+
                          (neural-network-bl bp-gd)
                          (neural-network-bl bp-gd-acc))))))

     ]
    ))

  (let* ((wl (reverse (neural-network-wl nn)))
         (bl (reverse (neural-network-bl nn)))
         (grad-nn (dcost-rec nn input output wl
                             (neural-network
                              (make-zero-mat-list wl)
                              (make-zero-mat-list bl)))))
      
    (nn-map
     (lambda ([m : Mat]) : Mat
       (matrix-scale m (/ 1 (length input)))) grad-nn)))


(define cmd-line
  (command-line
   #:program "deep.rkt"
   #:args (iter . arch)
   (cons iter arch)))

(: iter-count Integer)
(define iter-count
  (let ((arg (car cmd-line)))
    (if (string? arg)
        (let ((num (string->number arg)))
          (if (exact-integer? num) num 0 ))
        0
        )))

(: learn (-> neural-network (Listof Mat) (Listof Mat) neural-network))
(define (learn nn in out)

  (: learn-rec (-> neural-network (Listof Mat) (Listof Mat) Integer
                   neural-network))
  (define (learn-rec nn in out i)
    (match i
      [0 nn]
      [_
       (let* ((trained-nn (dcost nn in out))
              (rate-nn (nn-map
                        (lambda ([m : Mat]) : Mat
                          (matrix-scale m train-rate)) trained-nn))
              (new-nn
               (neural-network
                (map matrix- (neural-network-wl nn)
                     (neural-network-wl rate-nn))
                (map matrix- (neural-network-bl nn)
                     (neural-network-bl rate-nn)))))
         
         (begin
           ;; (printf "W1 ~a \n" (matrix-ref (car (neural-network-wl trained-nn)) 1 0))
           
           (learn-rec new-nn in out (- i 1))))
       ]
      ))

  (learn-rec nn in out iter-count))

(: perform (-> neural-network (Listof Mat) (Listof Mat) Number))
(define (perform nn in out)
  (match in
    ['() 0]
    [_
     (begin
       (printf "Result: ~a; Expected: ~a\n"
               (car (nn-forward (car in) nn))
               (car out))
       (perform nn (cdr in) (cdr out)))
     ]
    ))

(: get-arch (-> (Listof Any) (Listof Integer)))
(define (get-arch cmd-args)
  (foldl
   (lambda ([u : (U Complex False)] [acc : (Listof Integer)])
           : (Listof Integer)
     (if (exact-integer? u)
         (cons u acc)
         acc))
   '()
   (map string->number
        (map (lambda ([arg : Any]) : String
               (if (string? arg)
                   arg
                   ""))
               cmd-args))))
              
(define main
  (let* (
         ;; (arch (reverse nn-arch))
         (arch (get-arch (cdr cmd-line)))
         (nn (make-nn arch))
         (trained-nn (learn nn input-data out-data))
         (grad (dcost nn input-data out-data))
         (fwd-tree
          (forward (car input-data)
                   (neural-network-wl nn)
                   (neural-network-bl nn)))
         )
    (begin
      ;; 0
      (printf "~a\n" (cost nn input-data out-data))
      (printf "~a\n" (cost trained-nn input-data out-data))
      (perform trained-nn input-data out-data)
      )))

;; (trace main)
