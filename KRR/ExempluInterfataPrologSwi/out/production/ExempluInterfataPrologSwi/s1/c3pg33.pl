kb([
    [ plus(zero, X, X) ],
    [ neg(plus(X, Y, Z)), plus(succ(X), Y, succ(Z)) ]
]).

query([
    plus( succ(succ(zero)),
          succ(succ(succ(zero))),
          U)
]).