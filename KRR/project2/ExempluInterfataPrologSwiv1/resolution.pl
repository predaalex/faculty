:- use_module(library(socket)).
:- initialization(main, main).

main([PortAtom, KbQuestionFile]) :-
    atom_number(PortAtom, Port),

    % Load file that contains kb/1 and query/1
    consult(KbQuestionFile),

    tcp_socket(Sock),
    tcp_connect(Sock, '127.0.0.1':Port),
    tcp_open_socket(Sock, In, Out),
    run_fol_check(In, Out),

    close(In),
    close(Out),
    halt.
main(_) :-
    writeln('No port argument given.'),
    halt(1).

run_fol_check(_In, Out) :-
    kb(KB),
    query(Q),
    format(Out, 'TRACE: KB = ~w~n', [KB]),
    format(Out, 'TRACE: Query = ~w~n', [Q]),
    resolution(KB, Q, Answer, Out),
    send_result(Answer, Out).

send_result(entailed, Out) :-
    format(Out, 'entailed~n', []),
    flush_output(Out).
send_result(not_entailed, Out) :-
    format(Out, 'not_entailed~n', []),
    flush_output(Out).

resolution(KB, QueryLit, Result, Out) :-
    negate(QueryLit, NegatedQueryClause),
    append(KB, [NegatedQueryClause], AllClauses),
    (   resolution_refutation(AllClauses, Out)
    ->  Result = entailed
    ;   Result = not_entailed
    ).

negate(neg(P), [P]) :- !.
negate(P, [neg(P)]).

resolution_refutation(Clauses, Out) :-
    resolution_loop(Clauses, [], Out).

resolution_loop(Clauses, _, Out) :-
    member([], Clauses),
    !.

resolution_loop(Clauses, Processed, Out) :-
    format(Out, 'TRACE: Current clauses: ~w~n', [Clauses]),
    flush_output(Out),
    findall(
        Resolvent,
        (
            select(C1, Clauses, Rest),
            member(C2, Rest),
            resolve_clauses(C1, C2, Resolvent, Out),

            \+ member(Resolvent, Clauses),
            \+ member(Resolvent, Processed)
        ),
        NewResolvents
    ),
    (   NewResolvents = []
    ->  % No new clauses, can't derive empty clause -> satisfiable
        fail
    ;   append(Clauses, NewResolvents, Clauses1),
        append(Processed, NewResolvents, Processed1),
        resolution_loop(Clauses1, Processed1, Out)
    ).

resolve_clauses(Clause1, Clause2, Resolvent, Out) :-
    copy_term((Clause1, Clause2), (C1, C2)),
    select(L1, C1, Rest1),
    select(L2, C2, Rest2),
    complementary(L1, L2),
    append(Rest1, Rest2, Combined),
    sort(Combined, Resolvent).

complementary(neg(P), Q) :-
    \+ P = neg(_),
    P = Q.
complementary(P, neg(Q)) :-
    \+ P = neg(_),
    P = Q.
