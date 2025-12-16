:- use_module(library(socket)).

% Entry point: swipl -s resolution.pl -- 5004 kbq.pl
:- initialization(main, main).

% ---------- SOCKET / MAIN ----------

main([PortAtom, KbQuestionFile]) :-
    atom_number(PortAtom, Port),

    % Load file that contains kb/1 and query/1
    consult(KbQuestionFile),

    % Connect to Java
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

% ---------- RESOLUTION FRONT-END ----------

% resolution(+KBClauses, +QueryLiteral, -Result, +Out)
% Result = entailed | not_entailed

resolution(KB, QueryLit, Result, Out) :-
    negate(QueryLit, NegatedQueryClause),
    append(KB, [NegatedQueryClause], AllClauses),
    format(Out, 'TRACE: Added ¬Q as clause: ~w~n', [NegatedQueryClause]),
    format(Out, 'TRACE: Initial clause set S0: ~w~n', [AllClauses]),
    flush_output(Out),
    (   resolution_refutation(AllClauses, Out)
    ->  Result = entailed
    ;   Result = not_entailed
    ).

% Negate a literal to a single clause [Lit']:
% if Q = ¬P, then ¬Q = P
% if Q = P, then ¬Q = ¬P
negate(neg(P), [P]) :- !.
negate(P, [neg(P)]).

% ---------- RESOLUTION CORE ----------

% resolution_refutation(+Clauses, +Out)
% True if Clauses are unsatisfiable (empty clause is derivable)

resolution_refutation(Clauses, Out) :-
    resolution_loop(Clauses, [], Out).

% If we have the empty clause [], we derived a contradiction.
resolution_loop(Clauses, _, Out) :-
    member([], Clauses),
    format(Out, 'TRACE: Empty clause [] found. Refutation succeeded.~n', []),
    flush_output(Out),
    !.
resolution_loop(Clauses, Processed, Out) :-
    format(Out, 'TRACE: Current clauses: ~w~n', [Clauses]),
    flush_output(Out),
    % Generate all new resolvents from pairs of clauses
    findall(
        Resolvent,
        (
            select(C1, Clauses, Rest),
            member(C2, Rest),
            resolve_clauses(C1, C2, Resolvent, Out),
            % Skip resolvent if we already know it
            \+ member(Resolvent, Clauses),
            \+ member(Resolvent, Processed)
        ),
        NewResolvents
    ),
    (   NewResolvents = []
    ->  % No new clauses, can't derive empty clause -> satisfiable
        format(Out, 'TRACE: No new resolvents. Closure reached, no empty clause.~n', []),
        flush_output(Out),
        fail
    ;   format(Out, 'TRACE: New resolvents: ~w~n', [NewResolvents]),
        flush_output(Out),
        append(Clauses, NewResolvents, Clauses1),
        append(Processed, NewResolvents, Processed1),
        resolution_loop(Clauses1, Processed1, Out)
    ).

% ---------- RESOLUTION STEP ON TWO CLAUSES ----------

% resolve_clauses(+Clause1, +Clause2, -Resolvent, +Out)
% Clause = list of literals, literal = Pred(...) or neg(Pred(...))

resolve_clauses(Clause1, Clause2, Resolvent, Out) :-
    % Standardize variables apart by copying
    copy_term((Clause1, Clause2), (C1, C2)),
    % Pick a literal from each and try to resolve
    select(L1, C1, Rest1),
    select(L2, C2, Rest2),
    complementary(L1, L2),
    append(Rest1, Rest2, Combined),
    sort(Combined, Resolvent),  % sort to canonical form (no duplicates order)
    format(Out,
           'TRACE: Resolving ~w and ~w using ~w <-> ~w gives resolvent ~w~n',
           [Clause1, Clause2, L1, L2, Resolvent]),
    flush_output(Out).

% Literals are complementary if one is neg(P) and the other is P.

complementary(neg(P), Q) :-
    \+ P = neg(_),
    P = Q.
complementary(P, neg(Q)) :-
    \+ P = neg(_),
    P = Q.

% ---------- (END OF FILE) ----------
