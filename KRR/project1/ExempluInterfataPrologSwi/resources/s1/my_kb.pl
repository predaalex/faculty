kb([
    [neg(cat(X)),       animal(X)],
    [neg(animal(X)),    neg(purrs(X)), friendly(X)],
    [neg(animal(X)),    friendly(X),   aggressive(X)],
    [cat(luxor)],
    [purrs(luxor)]
]).

query(friendly(luxor)).
