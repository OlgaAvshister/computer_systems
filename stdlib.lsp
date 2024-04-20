(defun doreadstring (s i) (do
    (define c (readchar))
    (setchar s i c)
    (if (= c 0)
        s
        (doreadstring s (+ i 1)))))

(defun readstring (s)
    (doreadstring s 0))

(defun doprintstring (s i) (do
    (define c (getchar s i))
    (if (= c 0)
        i
        (do (printchar c)
            (doprintstring s (+ i 1))))))

(defun printstring (s) (doprintstring s 0))

(defun writedigits (num buf i)
    (if (!= num 0)
        (do (define digit (% num 10))
            (setchar buf i (+ digit 48))
            (writedigits (/ num 10) buf (- i 1)))
        (+ i 1)))

(defun printnumber (num)
    (if (= num 0)
        (printstring "0")
        (do
           (define buf (makestring 10))
           (if (> num 0)
               (do
                   (define i (writedigits num buf 9))
                   (printstring (+ buf i)))
               (if (= num -2147483648)
                   (printstring "-2147483648")
                   (do
                       (printstring "-")
                       (define i (writedigits (* num -1) buf 9))
                       (printstring (+ buf i))))))))
