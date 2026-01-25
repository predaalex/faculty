:- use_module(library(socket)).
:- use_module(library(readutil)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- dynamic fact/1.
:- dynamic rule_impl/2.

:- initialization(main, main).

main(Argv) :-
    ( Argv = [PortAtom, ScenarioPathAtom | _] ->
        atom_number(PortAtom, Port),
        atom_string(ScenarioPathAtom, ScenarioPath),
        run_client(Port, ScenarioPath)
    ; format(user_error, "Usage: swipl -s engine.pl -- <PORT> <SCENARIO_FILE>~n", []),
      halt(2)
    ).

run_client(Port, ScenarioPath) :-
    tcp_socket(Sock),
    tcp_connect(Sock, '127.0.0.1':Port),
    tcp_open_socket(Sock, In, Out),
    handle_session(In, Out, ScenarioPath),
    close(In),
    close(Out).


handle_session(In, Out, ScenarioPath) :-
    % -------- read + parse scenario --------
    read_scenario(ScenarioPath, RuleLines, GoalTerm, ComparisonTerms0),
    maplist(parse_rule_line, RuleLines, ParsedRules),          % rule(Premises, Head)
    rules_to_horn_kb(ParsedRules, HornKB),
    sort(ComparisonTerms0, ComparisonTerms),

    % -------- read answers from Java --------
    retractall(fact(_)),
    retractall(rule_impl(_,_)),
    read_answers(In, Values, BoolFacts0),
    derive_comparison_facts(ComparisonTerms, Values, CmpFacts),
    append(BoolFacts0, CmpFacts, Facts0),
    sort(Facts0, Facts),

    % -------- FIRST APPROACH: your algorithms --------
    ( forward_chaining(HornKB, Facts, GoalTerm) -> FC=entailed ; FC=not_entailed ),
    ( backward_chaining_v1(HornKB, Facts, GoalTerm) -> BC1=entailed ; BC1=not_entailed ),

    % -------- SECOND APPROACH: Prolog built-in BC, but dynamic --------
    assert_kb_as_prolog(ParsedRules),
    assert_facts_as_prolog(Facts),
    ( prove_goal_builtin(GoalTerm) -> PBC=entailed ; PBC=not_entailed ),

    % -------- debug prints (helpful in oral) --------
    format(Out, "Facts: ~w~n", [Facts]),
    format(Out, "Goal: ~w~n", [GoalTerm]),
    format(Out, "Forward chaining: ~w~n", [FC]),
    format(Out, "Backward chaining v1: ~w~n", [BC1]),
    format(Out, "Prolog built-in BC: ~w~n", [PBC]),
    format(Out, "~w~n", [FC]),
    flush_output(Out).

% ============================================================
% SCENARIO FILE READER
% Format:
% Rules:
% -If <cond> and <cond> then <cond>.
% Questions:
% -[id] ...
% The goal:
% -<cond>.
%
% cond grammar supports:
%   has(atom)
%   gt(id, number)
%   ge(id, number)
% ============================================================

read_scenario(Path, RuleLines, GoalTerm, ComparisonTerms) :-
    read_file_to_string(Path, S, []),
    split_string(S, "\n", "\r", Lines0),
    maplist(string_trim, Lines0, Lines),
    extract_rules(Lines, RuleLines),
    extract_goal(Lines, GoalStr),
    parse_condition_string(GoalStr, GoalTerm),
    extract_comparisons(RuleLines, ComparisonTerms).

string_trim(In, Out) :-
    normalize_space(string(Out), In).

extract_rules(Lines, RuleLines) :-
    drop_until("Rules:", Lines, AfterRules),
    take_until_any(["Questions:", "The goal:"], AfterRules, RuleSection),
    include(is_bullet, RuleSection, Bullets),
    maplist(strip_bullet, Bullets, RuleLines).

extract_goal(Lines, GoalStr) :-
    drop_until("The goal:", Lines, AfterGoal),
    include(is_bullet, AfterGoal, [First|_]),
    strip_bullet(First, GoalStr).

is_bullet(Line) :-
    string_length(Line, L), L > 1,
    sub_string(Line, 0, 1, _, "-").

strip_bullet(Line, Out) :-
    sub_string(Line, 1, _, 0, Out0),
    string_trim(Out0, Out).

drop_until(_, [], []).
drop_until(Marker, [Marker|Rest], Rest) :- !.
drop_until(Marker, [_|Rest], Out) :- drop_until(Marker, Rest, Out).

take_until_any(_, [], []).
take_until_any(Markers, [H|_], []) :- member(H, Markers), !.
take_until_any(Markers, [H|T], [H|Out]) :- take_until_any(Markers, T, Out).

% Extract comparison terms used in rules (gt/ge) so we know what to evaluate from numbers
extract_comparisons(RuleLines, ComparisonTerms) :-
    findall(Cmp,
        ( member(Line, RuleLines),
          parse_rule_line(Line, rule(Ps,_)),
          member(Cmp, Ps),
          is_comparison(Cmp)
        ),
        Cmps),
    sort(Cmps, ComparisonTerms).

is_comparison(gt(_, _)).
is_comparison(ge(_, _)).

% ============================================================
% KB "GRAMMAR": parse rule lines (restricted)
% "If <cond> then <cond>."
% "If <cond> and <cond> then <cond>."
% cond = has(atom) | gt(id,num) | ge(id,num)
% ============================================================

parse_rule_line(Line, rule(Premises, Head)) :-
    string_trim(Line, L1),
    ensure_period(L1, L2),
    % remove leading "If "
    ( sub_string(L2, 0, _, _, "If ") -> sub_string(L2, 3, _, 0, Body)
    ; throw(bad_rule_missing_if(Line))
    ),
    split_on_then(Body, Left, Right),
    split_on_and(Left, PremStrs),
    maplist(parse_condition_string, PremStrs, Premises),
    parse_condition_string(Right, Head).

ensure_period(Line, Out) :-
    ( sub_string(Line, _, 1, 0, ".") -> Out = Line
    ; string_concat(Line, ".", Out)
    ).

split_on_then(S, Left, Right) :-
    ( sub_string(S, ThenPos, _, After, " then ") ->
        sub_string(S, 0, ThenPos, _, Left0),
        sub_string(S, _, After, 0, Right0),
        string_trim(Left0, Left),
        string_trim(Right0, Right)
    ; throw(bad_rule_missing_then(S))
    ).

split_on_and(S, PartsTrimmed) :-
    split_on_substring(" and ", S, Parts0),
    maplist(string_trim, Parts0, PartsTrimmed).

split_on_substring(Delim, S, [Left|Rest]) :-
    ( sub_string(S, Pos, _, After, Delim) ->
        sub_string(S, 0, Pos, _, Left0),
        sub_string(S, _, After, 0, Right0),
        string_trim(Left0, Left),
        split_on_substring(Delim, Right0, Rest)
    ; string_trim(S, Left),
      Rest = []
    ).

parse_condition_string(S0, Term) :-
    string_trim(S0, S1),
    remove_trailing_period(S1, S),
    string_codes(S, Cs),
    phrase(condition(Term), Cs), !.
parse_condition_string(S, _) :-
    throw(bad_condition(S)).

remove_trailing_period(S, Out) :-
    ( sub_string(S, _, 1, 0, ".") ->
        sub_string(S, 0, _, 1, Out)
    ; Out = S
    ).

condition(has(A)) --> "has(", atom_name(A), ")".
condition(gt(A,N)) --> "gt(", atom_name(A), ",", ws0, number_string(N), ")".
condition(ge(A,N)) --> "ge(", atom_name(A), ",", ws0, number_string(N), ")".

ws0 --> [C], { code_type(C, space) }, !, ws0.
ws0 --> [].

atom_name(A) --> atom_chars(Cs), { atom_codes(A, Cs) }.
atom_chars([C|Cs]) --> [C], { valid_atom_char(C) }, !, atom_chars(Cs).
atom_chars([]) --> [].
valid_atom_char(C) :- code_type(C, alnum) ; C=:=0'_; C=:=0'-.

number_string(N) --> number_chars(Cs), { number_codes(N, Cs) }.
number_chars([C|Cs]) --> [C], { (C>=0'0, C=<0'9) ; C=:=0'. ; C=:=0'- }, !, number_chars(Cs).
number_chars([]) --> [].

% ============================================================
% Horn KB: clause is [n(P1), n(P2), Head]
% Premises/Head are ground terms like has(x) or gt(id,38)
% ============================================================

rules_to_horn_kb(ParsedRules, HornKB) :-
    maplist(rule_to_clause, ParsedRules, HornKB).

rule_to_clause(rule(Premises, Head), Clause) :-
    maplist(neg_lit, Premises, NegPremises),
    append(NegPremises, [Head], Clause).

neg_lit(P, n(P)).

% ============================================================
% Read answers from Java:
%   ans:<id>=<value>
% Numbers -> value(Id, Num)
% Booleans -> has(Id) when yes/true/1
% ============================================================

read_answers(In, Values, BoolFacts) :-
    read_line_to_string(In, Line0),
    ( Line0 == end_of_file -> Values=[], BoolFacts=[]
    ; string_trim(Line0, Line),
      ( Line = "done" -> Values=[], BoolFacts=[]
      ; parse_answer_line(Line, Values1, BoolFacts1),
        read_answers(In, ValuesRest, BoolFactsRest),
        append(Values1, ValuesRest, Values),
        append(BoolFacts1, BoolFactsRest, BoolFacts)
      )
    ).

parse_answer_line(Line, Values, BoolFacts) :-
    ( sub_string(Line, 0, _, _, "ans:") ->
        sub_string(Line, 4, _, 0, Rest),
        ( sub_string(Rest, EqPos, _, After, "=") ->
            sub_string(Rest, 0, EqPos, _, IdStr0),
            sub_string(Rest, _, After, 0, ValStr0),
            string_trim(IdStr0, IdStr),
            string_trim(ValStr0, ValStr),
            atom_string(Id, IdStr),
            classify_value(Id, ValStr, Values, BoolFacts)
        ; throw(bad_answer(Line))
        )
    ; throw(bad_answer(Line))
    ).

classify_value(Id, ValStr, [value(Id, Num)], []) :-
    catch(number_string(Num, ValStr), _, fail), !.
classify_value(Id, ValStr, [], [has(Id)]) :-
    string_lower(ValStr, L),
    ( L = "yes" ; L = "true" ; L = "1" ), !.
classify_value(_Id, _ValStr, [], []) :-
    true.

% Evaluate comparison terms (gt/ge) from numeric values:
derive_comparison_facts(ComparisonTerms, Values, Facts) :-
    findall(Cmp,
        ( member(Cmp, ComparisonTerms),
          comparison_holds(Cmp, Values)
        ),
        Facts0),
    sort(Facts0, Facts).

comparison_holds(gt(Id, K), Values) :-
    member(value(Id, V), Values),
    V > K.
comparison_holds(ge(Id, K), Values) :-
    member(value(Id, V), Values),
    V >= K.

% ============================================================
% Forward chaining
% ============================================================

forward_chaining(HornKB, Facts, Goal) :-
    closure(HornKB, Facts, Closure),
    member(Goal, Closure).

closure(HornKB, Facts, Closure) :-
    forward_step(HornKB, Facts, Facts1),
    ( Facts1 == Facts -> Closure = Facts
    ; closure(HornKB, Facts1, Closure)
    ).

forward_step(HornKB, Facts, FactsOut) :-
    findall(Head,
        ( member(Clause, HornKB),
          clause_head(Clause, Head),
          \+ member(Head, Facts),
          clause_premises(Clause, Premises),
          subset_list(Premises, Facts)
        ),
        NewHeads0),
    sort(NewHeads0, NewHeads),
    append(Facts, NewHeads, Facts1),
    sort(Facts1, FactsOut).

clause_head(Clause, Head) :- last(Clause, Head).

clause_premises(Clause, Premises) :-
    append(NegPremises, [_Head], Clause),
    maplist(unwrap_neg, NegPremises, Premises).

unwrap_neg(n(P), P).

subset_list([], _).
subset_list([X|Xs], Set) :- member(X, Set), subset_list(Xs, Set).

% ============================================================
% Backward chaining v1 (DFS)
% ============================================================

backward_chaining_v1(HornKB, Facts, Goal) :-
    bc1_prove(HornKB, Facts, Goal, []).

bc1_prove(_KB, Facts, Goal, _Visited) :-
    member(Goal, Facts), !.
bc1_prove(KB, Facts, Goal, Visited) :-
    \+ member(Goal, Visited),
    member(Clause, KB),
    clause_head(Clause, Goal),
    clause_premises(Clause, Premises),
    bc1_all(KB, Facts, Premises, [Goal|Visited]).

bc1_all(_KB, _Facts, [], _Visited).
bc1_all(KB, Facts, [P|Ps], Visited) :-
    bc1_prove(KB, Facts, P, Visited),
    bc1_all(KB, Facts, Ps, Visited).

% ============================================================
% SECOND APPROACH: Dynamic Prolog backward chaining (no hardcoding)
% We assert generic predicates:
%   fact(Term).
%   rule_impl(Head, PremisesList).
% and prove using Prolog resolution style:
%   prove(Goal) :- fact(Goal).
%   prove(Goal) :- rule_impl(Goal, Ps), prove_all(Ps).
%
% This is "Prolog backward chaining", but built from file dynamically.
% ============================================================

assert_kb_as_prolog(ParsedRules) :-
    retractall(rule_impl(_,_)),
    forall(member(rule(Ps,H), ParsedRules),
           assertz(rule_impl(H, Ps))).

assert_facts_as_prolog(Facts) :-
    retractall(fact(_)),
    forall(member(F, Facts),
           assertz(fact(F))).

prove_goal_builtin(Goal) :-
    prove(Goal).

prove(Goal) :-
    fact(Goal), !.
prove(Goal) :-
    rule_impl(Goal, Premises),
    prove_all(Premises).

prove_all([]).
prove_all([P|Ps]) :-
    prove(P),
    prove_all(Ps).
