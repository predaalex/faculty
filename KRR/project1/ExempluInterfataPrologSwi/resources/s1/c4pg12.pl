kb([
    [neg(hound(X)), howl(X)],
    [neg(have(X,Y)), neg(cat(Y)), neg(have(X,Z)), neg(mouse(Z))],
    [neg(ls(X)), neg(have(X,Y)), neg(howl(Y))],
    [have(john,A)],
    [cat(A), hound(A)]
]).

query([ls(john), have(john,X), mouse(X)]).
