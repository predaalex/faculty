kb([
    [on(A,B)],
    [on(B,C)],
    [green(A)],
    [neg(green(C))]
]).

query([green(X), neg(green(Y)), on(X,Y)]).
