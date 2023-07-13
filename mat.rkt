#!/bin/env racket
#lang typed/racket

(require math)
(require math/matrix)

(: nn-map (-> (-> (Listof Mat) Mat)
              neural-network neural-network))
(define (nn-map proc . nn)
    (neural-network
     (map proc
          (foldl
           (lambda ([nn : neural-network]
                    [mat-list : (Listof (Listof Mat))]) : (Listof (Listof Mat))
             (cons (neural-network-wl nn) mat-list)) '() nn))
     (map proc
          (foldl
           (lambda ([nn : neural-network]
                    [mat-list : (Listof (Listof Mat))]) : (Listof (Listof Mat))
             (cons (neural-network-bl nn) mat-list)) '() nn))))


