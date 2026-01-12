:- use_module(library(socket)).
:- initialization(main, main).

main([PortAtom, FileClauses]) :-
    atom_number(PortAtom, Port),
    consult(FileClauses),
    tcp_socket(Socket),
    tcp_connect(Socket, '127.0.0.1':Port),
    tcp_open_socket(Socket, In, Out),
    sat_solver(Out),
    close(In),
    close(Out),
    halt.

main(_) :-
    writeln('Missing arguments.'),
    halt(1).

sat_solver(Out) :-
    clauses(ClausesList),
    (   dpll(ClausesList, [], Model)
    ->  format(Out, 'YES~n', []),
        print_model(Out, Model)
    ;   format(Out, 'NOT~n', [])
    ),
    flush_output(Out).

dpll(Clauses, AssignIn, AssignOut) :-
    unit_clause_propagation(Clauses, C1, UnitAsgn),
    pure_literal_propagation(C1, C2, PureAsgn),
    append(UnitAsgn, PureAsgn, NewAsgn),
    append(NewAsgn, AssignIn, AssignMid),
    (   C2 = []
    ->  AssignOut = AssignMid
    ;   member([], C2)
    ->  fail
    ;   choose_literal(C2, Lit),
        (   propagate_literal(Lit, C2, CTrue),
            dpll(CTrue, [Lit|AssignMid], AssignOut)
        ;   complement(Lit, Comp),
            propagate_literal(Comp, C2, CFalse),
            dpll(CFalse, [Comp|AssignMid], AssignOut)
        )
    ).

choose_literal([[L|_]|_], L).

unit_clause_propagation(Clauses, FinalClauses, Assignments) :-
    unit_clause_propagation_(Clauses, [], FinalClauses, Assignments).

unit_clause_propagation_(Clauses, Acc, FinalClauses, Assignments) :-
    (   first_unit_clause(Clauses, Lit, RestClauses)
    ->  propagate_literal(Lit, RestClauses, PropagatedClauses),
        unit_clause_propagation_(PropagatedClauses, [Lit|Acc],
                                 FinalClauses, Assignments)
    ;   FinalClauses = Clauses,
        reverse(Acc, Assignments)
    ).

first_unit_clause(Clauses, Lit, RestClauses) :-
    select([Lit], Clauses, RestClauses),
    !.

complement(neg(X), X) :- !.
complement(X, neg(X)).

simplify_clause(Lit, Clause, true) :-
    member(Lit, Clause),
    !.
simplify_clause(Lit, Clause, reduced(NewClause)) :-
    complement(Lit, Comp),
    select(Comp, Clause, NewClause),
    !.
simplify_clause(_, Clause, unchanged(Clause)).

propagate_literal(Lit, Clauses, NewClauses) :-
    findall(C2,
        ( member(C, Clauses),
          simplify_clause(Lit, C, Res),
          ( Res = true            -> fail
          ; Res = reduced(C2)
          ; Res = unchanged(C2)
          )
        ),
        NewClauses).

pure_literal_propagation(Clauses, FinalClauses, PureAssignments) :-
    find_pure_literals(Clauses, PureAssignments),
    propagate_pure_literals(PureAssignments, Clauses, FinalClauses).

find_pure_literals(Clauses, PureLits) :-
    collect_literals(Clauses, AllLits),
    classify_polarities(AllLits, PosBases, NegBases),
    findall(Lit,
        (
            member(Base, PosBases),
            \+ member(Base, NegBases),
            Lit = Base
        ;
            member(Base, NegBases),
            \+ member(Base, PosBases),
            Lit = neg(Base)
        ),
        PureLits).

collect_literals(Clauses, Literals) :-
    findall(L,
        ( member(C, Clauses),
          member(L, C)
        ),
        Literals).

classify_polarities(Lits, PosBases, NegBases) :-
    classify_polarities(Lits, [], [], Pos0, Neg0),
    sort(Pos0, PosBases),
    sort(Neg0, NegBases).

classify_polarities([], PosAcc, NegAcc, PosAcc, NegAcc).
classify_polarities([neg(X)|Rest], PosAcc, NegAcc, PosBases, NegBases) :-
    !,
    classify_polarities(Rest, PosAcc, [X|NegAcc], PosBases, NegBases).
classify_polarities([X|Rest], PosAcc, NegAcc, PosBases, NegBases) :-
    classify_polarities(Rest, [X|PosAcc], NegAcc, PosBases, NegBases).

propagate_pure_literals([], Clauses, Clauses).
propagate_pure_literals([Lit|Rest], Clauses, FinalClauses) :-
    propagate_literal(Lit, Clauses, Clauses1),
    propagate_pure_literals(Rest, Clauses1, FinalClauses).

print_model(Out, Model) :-
    reverse(Model, RevModel),
    format(Out, '{', []),
    print_model_list(Out, RevModel),
    format(Out, '}~n', []).

print_model_list(_, []).
print_model_list(Out, [Lit]) :-
    literal_to_pair(Lit, V, Val),
    format(Out, '~w/~w', [V, Val]).
print_model_list(Out, [Lit|Ls]) :-
    literal_to_pair(Lit, V, Val),
    format(Out, '~w/~w; ', [V, Val]),
    print_model_list(Out, Ls).

literal_to_pair(neg(X), X, false) :- !.
literal_to_pair(X, X, true).
