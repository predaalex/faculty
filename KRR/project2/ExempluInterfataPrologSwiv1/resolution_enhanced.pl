:- use_module(library(socket)).
:- use_module(library(apply)).  % for maplist
:- use_module(library(lists)).

% Entry point: swipl -s main.pl -- 5004 kb_file.pl
:- initialization(main, main).

% ---------- SOCKET / MAIN ----------

main([PortAtom, KbQuestionFile]) :-
    atom_number(PortAtom, Port),
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
    resolution(KB, Q, Answer),
    send_result(Answer, Out).

send_result(entailed, Out) :-
    format(Out, 'entailed~n', []),
    flush_output(Out).
send_result(not_entailed, Out) :-
    format(Out, 'not_entailed~n', []),
    flush_output(Out).

% ---------- RESOLUTION FRONT-END ----------

resolution(KB, QueryLits, Result) :-
    negate_clause(QueryLits, NegatedQueryClause),
    append(KB, [NegatedQueryClause], AllClauses),
    (   resolution_refutation(AllClauses)
    ->  Result = entailed
    ;   Result = not_entailed
    ).

% Negate each literal in the query clause
negate_clause([], []).
negate_clause([neg(L)|Rest], [L1|R]) :-
    L1 = L,
    negate_clause(Rest, R).
negate_clause([L|Rest], [neg(L)|R]) :-
    \+ L = neg(_),
    negate_clause(Rest, R).

% ---------- RESOLUTION CORE ----------

resolution_refutation(Clauses) :-
    resolution_loop(Clauses, []).

resolution_loop(Clauses, _) :-
    member([], Clauses), !.  % empty clause ⇒ contradiction
resolution_loop(Clauses, Processed) :-
    findall(
        Resolvent,
        (
            select(C1, Clauses, Rest),
            member(C2, Rest),
            resolve_clauses(C1, C2, Resolvent),
            \+ member(Resolvent, Clauses),
            \+ member(Resolvent, Processed)
        ),
        NewResolvents
    ),
    (   NewResolvents = []
    ->  fail
    ;   append(Clauses, NewResolvents, Clauses1),
        append(Processed, NewResolvents, Processed1),
        resolution_loop(Clauses1, Processed1)
    ).

% ---------- RESOLUTION STEP WITH SUBSTITUTION ----------

resolve_clauses(C1, C2, Resolvent) :-
    copy_term((C1, C2), (Clause1, Clause2)),
    select(L1, Clause1, Rest1),
    select(L2, Clause2, Rest2),
    complementary(L1, L2, Subst),
    % apply substitution to remaining literals
    apply_substitution(Rest1, Subst, NewRest1),
    apply_substitution(Rest2, Subst, NewRest2),
    append(NewRest1, NewRest2, Combined),
    sort(Combined, Resolvent).

% Determine if two literals are complementary and return the MGU (substitution)
complementary(neg(P), Q, Subst) :-
    \+ P = neg(_),
    unifiable(P, Q, Subst).
complementary(P, neg(Q), Subst) :-
    \+ P = neg(_),
    unifiable(P, Q, Subst).

% Apply a substitution to a list of literals
apply_substitution(Literals, Subst, Result) :-
    maplist(apply_one_subst(Subst), Literals, Result).

apply_one_subst(Subst, Literal, NewLiteral) :-
    copy_term(Subst+Literal, SubstCopy+LiteralCopy),
    bind_subst(SubstCopy),
    NewLiteral = LiteralCopy.

% bind_subst applies the substitution as variable bindings
bind_subst([]).
bind_subst([Var=Term|Rest]) :-
    Var = Term,
    bind_subst(Rest).
