#!/usr/bin/env racket

#lang typed/racket

#|
(require/typed math/matrix
  [#:opaque Matrix matrix?]
  [matrix+ (-> (Matrix Number) (Matrix Number) (Matrix Number))]
  [matrix- (-> (Matrix Number) (Matrix Number) (Matrix Number))]
  [matrix* (-> (Matrix Number) (Matrix Number) (Matrix Number))]
  [matrix-map (-> (-> Number Number) (Matrix Number) (Matrix Number))]
  [make-matrix (-> Integer Integer Number (Matrix Number))]
  [list->matrix (-> Integer Integer (Listof Number) (Matrix Number))]
  [matrix->list (-> (Matrix Number) (Listof Number))]
  [matrix-num-rows (-> (Matrix Number) Index)]
  [matrix-num-cols (-> (Matrix Number) Index)]
  [matrix-set-col (-> (Matrix Number) Integer (Matrix Number) (Matrix Number))]
  [matrix-set-row (-> (Matrix Number) Integer (Matrix Number) (Matrix Number))]
  [matrix-row (-> (Matrix Number) Integer (Matrix Number))]
  [matrix-col (-> (Matrix Number) Integer (Matrix Number))]
  [matrix-scale (-> (Matrix Number) Number (Matrix Number))] 
  )
|#

(require math/matrix)

(provide train
         perform
         learn
         make-nn
         print-nn
         cost
         )

(define-type Mat (Matrix Real))

(define-type Weight Real)
(define-type Bias Real)

(define-type TrainSample (Pair Mat Mat))
(define-type TrainData (Listof TrainSample))

(define-type WeightList (Listof (Matrix Weight)))
(define-type BiasList (Listof (Matrix Bias)))

(: train-data-get-in (-> TrainData Mat))
(define (train-data-get-in train-data)
  (cdar train-data))

(: train-data-get-out (-> TrainData Mat))
(define (train-data-get-out train-data)
  (caar train-data))

(struct neural-network
  (
   [wl : (Listof Mat)]
   [bl : (Listof Mat)]))

(: print-nn (-> neural-network Void))
(define (print-nn nn)
  (begin 
    (printf "\nNN Weights:\n")
    (map mat-print (neural-network-wl nn))

    (printf "\nNN Biases:\n")
    (map mat-print (neural-network-bl nn))
    (printf "\n")
    (void)))

(struct feed-forward
  (
   [res : (Listof Mat)]
   [weights : WeightList]
   [biases : BiasList]
   ))

(struct backprop-nn
  (
   [wmat-list : (Listof (Matrix Weight))]
   [bmat-list : (Listof (Matrix Bias))]
   ))

(struct backprop-layer
  (
   [pd-prev : (Vectorof Real)]
   [w-mat : (Matrix Weight)]
   [b-mat : (Matrix Bias)]
   ))

(struct backprop-neuron
  (
   [w-list : (Vectorof Weight)]
   [pd-prev : (Vectorof Real)]
   ))

(: sigmoid (-> Real Real))
(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(: nn-apply (-> (-> Mat Mat Mat) neural-network neural-network neural-network))
(define (nn-apply proc nn1 nn2)
  (neural-network
   (map proc
        (neural-network-wl nn1)
        (neural-network-wl nn2))
    (map proc
        (neural-network-bl nn1)
        (neural-network-bl nn2))))

(: mat-print (-> Mat Void))
(define (mat-print mat)

  (: mat-print-rec (-> (Listof Mat) Void))
  (define (mat-print-rec rows)
    (match rows
      ['() (void)]
      [_
       (printf "\t~a\n" (car rows))
       (mat-print-rec (cdr rows))
       ]
      ))
  
  (let ((rows (matrix-rows mat)))
    (mat-print-rec rows)))
    
 
(: forward-layer (-> Mat Mat Mat Mat))
(define (forward-layer in w b)
  (matrix-map sigmoid (matrix+ (matrix* in w) b)))

(: forward (-> Mat WeightList BiasList feed-forward))
(define (forward in wl bl)

  (: forward-acc (-> Mat WeightList BiasList feed-forward feed-forward))
  (define (forward-acc in wl bl ff-acc)
    (match wl
        ['() ff-acc]
        [l
         (let* ((w-mat (car wl))
                (b-mat (car bl))
                (layer-activation (forward-layer in (car wl) (car bl)))
                (ff
                 (feed-forward
                  (cons layer-activation (feed-forward-res ff-acc))
                  (cons w-mat (feed-forward-weights ff-acc))
                  (cons b-mat (feed-forward-biases ff-acc)))))
                  
           (forward-acc layer-activation (cdr wl) (cdr bl) ff))
        ]
        ))

  (forward-acc in wl bl
               (feed-forward
                (list in)
                (list)
                (list))))

(: nn-forward (-> Mat neural-network feed-forward))
(define (nn-forward in nn)
  (forward in
           (neural-network-wl nn)
           (neural-network-bl nn)))

(: randf (-> Real Real))
(define (randf r)
  (- (* (random) 2) 1)
  )

(: make-random-mat (-> Integer Integer Mat))
(define (make-random-mat row col)
  (matrix-map randf
              (make-matrix row col 0)
              ))

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
  
  (let ((rev-arch (reverse arch)))
    (begin
      ;; (print (make-wl-rec rev-arch '()))
     (neural-network (make-wl-rec rev-arch '()) (make-bl-rec rev-arch '())))))

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

(: cost (-> neural-network TrainData Real))
(define (cost nn data)

  (: cost-rec (-> neural-network TrainData Real Real))
  (define (cost-rec nn data err)
  (match data
    ['() err]
    [l
     (let* ((ff (nn-forward (train-data-get-in data) nn))
            (nn-y (car (feed-forward-res ff)))
            (y (train-data-get-out data))
            (mat-diff (matrix- nn-y y))
            (diff (apply + (matrix->list mat-diff))))
       (cost-rec nn (cdr data) (+ err (* diff diff))))
     ]
    ))

  (/ (cost-rec nn data 0)
     (length data)))

(: grad-neuron
 (-> (Vectorof Real) Real Real (Vectorof Real) (Vectorof Real) Integer
     backprop-neuron))
(define (grad-neuron w-acc-vec diff-i ai ai-prev-vec
                      pd-prev-vec-acc i)

  (: grad-neuron-rec (-> Real Real (Vectorof Real) Integer Void))
  (define (grad-neuron-rec diff-i ai ai-prev-vec i)
    (match i
      [-1 (void)]
      [_
       (let* ((ai-prev (vector-ref ai-prev-vec i))
              (wi-pd (* 2 diff-i ai (- 1 ai) ai-prev))
              (last-pd-ai-prev (vector-ref pd-prev-vec-acc i))
              (cur-w (vector-ref w-acc-vec i))
              (pd-ai-prev (* 2 diff-i ai (- 1 ai) cur-w))
           )
         
         (begin
           ;; (printf "Diff ~a, \nai ~a, \nai-prev ~a \nwi ~a\n" diff-i ai ai-prev wi-pd)
           (vector-set! w-acc-vec i wi-pd)
           (vector-set! pd-prev-vec-acc i
                     (+ pd-ai-prev last-pd-ai-prev))
           (grad-neuron-rec
            diff-i
            ai
            ai-prev-vec
            (- i 1))))
       ]
      ))

  (begin
    (grad-neuron-rec diff-i ai ai-prev-vec i)
    (backprop-neuron w-acc-vec pd-prev-vec-acc)))

(: grad-layer
   (-> Index Index (Matrix Weight)
       (Listof Real) (Vectorof Real) (Listof Real)
       (Vectorof Real) (Vectorof Real) Integer backprop-layer))
(define (grad-layer w-row w-col w-mat b-acc-list prev-diff-vec ai-l
                     diff-vec ai-prev-vec i)
  (match ai-l
    ['() (backprop-layer
          prev-diff-vec
          w-mat
          (list->matrix 1 w-col b-acc-list))
         ]
    [_
     
     (let* ((ai (car ai-l))
            (diff (vector-ref diff-vec i))
            (cur-row-vec (matrix->vector (matrix-col w-mat i)))
            (neuron-bp
             (grad-neuron cur-row-vec diff ai ai-prev-vec
                           prev-diff-vec
                           (- w-row 1)))
            (neuron-grad (backprop-neuron-w-list neuron-bp))
            (neuron-diff-vec (backprop-neuron-pd-prev neuron-bp))
            (n-grad-mat (vector->matrix w-row 1 neuron-grad))
            (bias-gd (* 2 diff ai (- 1 ai))))
       
       (begin
         ;; (printf "Current activation: ~a\n" neuron-diff-l)
         (grad-layer w-row w-col (matrix-set-col w-mat i n-grad-mat)
                      (cons bias-gd b-acc-list) neuron-diff-vec
                      (cdr ai-l) diff-vec ai-prev-vec (+ i 1))))
       ]
    ))

(: grad-nn
   (-> (Listof Mat) (Listof Mat) (Listof Mat) (Listof Mat) (Vectorof Real)
       neural-network))
(define (grad-nn forward-list wmat-list
                  wgrad-mat-list-acc bgrad-mat-list-acc diff-vec)
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
             (grad-layer w-rows w-cols cur-wmat
                          '() (make-vector w-rows 0)
                          (matrix->list (car forward-list))
                          diff-vec
                          (matrix->vector (cadr forward-list))
                          0))
            (bp-prev-diff-vec (backprop-layer-pd-prev layer-bp))
            (bp-wgrad-mat (backprop-layer-w-mat layer-bp))
            (bp-bgrad-mat (backprop-layer-b-mat layer-bp))
            )
       
       (begin
         ;; (printf "BP grad ~a\n" wmat-list)
         ;; (mat-print bp-wgrad-mat)
         (grad-nn (cdr forward-list) (cdr wmat-list)
                   (cons bp-wgrad-mat wgrad-mat-list-acc)
                   (cons bp-bgrad-mat bgrad-mat-list-acc)
                   bp-prev-diff-vec)))
     ]
    ))

(: make-zero-mat-list (-> (Listof Mat) (Listof Mat)))
(define (make-zero-mat-list mat-list)
  (foldr (lambda ([mat : Mat] [mat-l : (Listof Mat)]) : (Listof Mat)
           (cons (make-matrix
                  (matrix-num-rows mat)
                  (matrix-num-cols mat)
                  0) mat-l)) '() mat-list))

(: grad (-> neural-network TrainData neural-network))
(define (grad nn data)

  (: grad-rec
     (-> neural-network TrainData neural-network neural-network))
  (define (grad-rec nn data bp-gd-acc)
  (match data
    ['() bp-gd-acc]
    [l
    (let* ((fwd-tree (nn-forward (train-data-get-in data) nn))
           (fwd-res (car (feed-forward-res fwd-tree)))
           (fwd-wl (feed-forward-weights fwd-tree))
           (expected-res (train-data-get-out data))
           (res-diff (matrix->vector (matrix- fwd-res expected-res)))
           (bp-gd (grad-nn (feed-forward-res fwd-tree) fwd-wl '() '() res-diff)))

      (begin
        ;; (printf "Samples ~a\n" (car input))
        (printf "Processed one sample\n")
        (grad-rec nn (cdr data)
                   (nn-apply matrix+ bp-gd bp-gd-acc))))
     ]
    ))

  (let* ((wl (neural-network-wl nn))
         (bl (neural-network-bl nn))
         (grad-nn (grad-rec nn data 
                             (neural-network
                              (make-zero-mat-list wl)
                              (make-zero-mat-list bl)))))
      
    (nn-map
     (lambda ([m : Mat]) : Mat
       (matrix-scale m (/ 1 (length data)))) grad-nn)))


(: learn (-> neural-network TrainData Real Integer
             neural-network))
(define (learn nn data rate iter)

  (: learn-rec (-> neural-network TrainData Integer
                   neural-network))
  (define (learn-rec nn data i)
    (match i
      [0 nn]
      [_
       (let* ((trained-nn (grad nn data))
              (rate-nn (nn-map
                        (lambda ([m : Mat]) : Mat
                          (matrix-scale m rate)) trained-nn))
              (new-nn (nn-apply matrix- nn rate-nn)))
         
         (begin
           ;; (printf "W1 ~a \n" (matrix-ref (car (neural-network-wl trained-nn)) 1 0))
           (learn-rec new-nn data (- i 1))))
       ]
      ))

  (learn-rec nn data iter))

(: perform (-> neural-network TrainData Number))
(define (perform nn data)
  (match data
    ['() 0]
    [_
     (begin
       (printf "NN's Result: ~a; \nExpected: ~a\n\n"
               (car (feed-forward-res (nn-forward (train-data-get-in data) nn)))
               (train-data-get-out data))
       (perform nn (cdr data)))
     ]
    ))


(: train (-> Integer (Listof Integer) TrainData Real
             neural-network))
(define (train iter-count arch data train-rate)
  (let*
      (
       (nn (make-nn arch))
       (trained-nn (learn nn data train-rate iter-count))
       )

    trained-nn
    ))

#|
(: parse-csv-train-data (-> String TrainData))
(define (parse-csv-train-data fname)
  (csv-map
   (lambda ([csv-row : (Listof String)]) : TrainSample
     (let ((real-list
            (map
             (lambda ([val : String]) : Real
               (let ((num (string->number val)))
                 (if (real? num) num 0)))
             csv-row)))
       (cons
        (make-matrix 1 1 (car real-list))
        (list->matrix 24 24 (cdr real-list)))))
   fname))
|#

(define xor-data
  (list 
    (cons (matrix [[0]]) (matrix [[0 0]]))
    (cons (matrix [[1]]) (matrix [[0 1]]))
    (cons (matrix [[1]]) (matrix [[1 0]]))
    (cons (matrix [[0]]) (matrix [[1 1]]))
    ))
