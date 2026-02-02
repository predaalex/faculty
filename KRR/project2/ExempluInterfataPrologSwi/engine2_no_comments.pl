:- use_module(library(socket)).
:- use_module(library(readutil)).
:- use_module(library(lists)).
:- use_module(library(dcg/basics)).

:- initialization(main, main).

main(Argv) :-
    ( Argv = [PortAtom, ScenarioPathAtom | _] ->
        atom_number(PortAtom, Port),
        atom_string(ScenarioPathAtom, ScenarioPath),
        run_client(Port, ScenarioPath)
    ; format(user_error, "Usage: swipl -s engine2.pl -- <PORT> <SCENARIO_FILE>~n", []),
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
    read_scenario(ScenarioPath, RuleLines, GoalVar),
    format(user_error, "[DEBUG] RuleLines = ~w~n", [RuleLines]),
    format(user_error, "[DEBUG] GoalVar = ~w~n", [GoalVar]),

    maplist(parse_fuzzy_rule_line, RuleLines, Rules0),
    format(user_error, "[DEBUG] Parsed Rules0 = ~w~n", [Rules0]),

    include(rule_matches_goal(GoalVar), Rules0, Rules),
    format(user_error, "[DEBUG] Filtered Rules = ~w~n", [Rules]),

    read_answers(In, Values),
    format(user_error, "[DEBUG] Input Values = ~w~n", [Values]),

    aggregate_output_curve(Rules, Values, Curve),
    format(user_error, "[DEBUG] Aggregated Curve = ~w~n", [Curve]),

    defuzzify_centroid(Curve, CrispOut),
    format(user_error, "[DEBUG] CrispOut = ~w~n", [CrispOut]),

    format(Out, "result:~w=~2f~n", [GoalVar, CrispOut]),
    flush_output(Out),
    format(user_error, "[DEBUG] Output flushed~n", []).



rule_matches_goal(GoalVar, rule(_Conn, _Ante, GoalVar/_Pred)).

read_scenario(Path, RuleLines, GoalVar) :-
    read_file_to_string(Path, S, []),
    split_string(S, "\n", "\r", Lines0),
    maplist(string_trim, Lines0, Lines),
    extract_section_lines("Rules:", Lines, RuleLines0),
    include(is_rule_line, RuleLines0, RuleLines),
    extract_goal_var(Lines, GoalVar).

is_rule_line(Line) :-
    Line \= "",
    sub_string(Line, 0, 1, _, "-").

string_trim(S0, S) :-
    normalize_space(string(S), S0).

extract_section_lines(Header, Lines, SectionLines) :-
    ( append(_, [Header|Rest], Lines) ->
        take_until_next_header(Rest, SectionLines)
    ; SectionLines = []
    ).

take_until_next_header([], []).
take_until_next_header([L|_], []) :-
    is_header(L), !.
take_until_next_header([L|Ls], [L|Out]) :-
    \+ is_header(L),
    take_until_next_header(Ls, Out).

is_header("Rules:")     :- !.
is_header("Questions:") :- !.
is_header("The goal:")  :- !.

extract_goal_var(Lines, GoalVar) :-
    ( append(_, ["The goal:"|Rest], Lines) ->
        first_nonempty(Rest, GoalLine0),
        strip_leading_dash(GoalLine0, GoalLine),
        string_lower(GoalLine, GoalLower),
        atom_string(GoalVar, GoalLower)
    ; throw(no_goal_found)
    ).

first_nonempty([L|_], L) :- L \= "", !.
first_nonempty([_|Ls], L) :- first_nonempty(Ls, L).
first_nonempty([], "").

strip_leading_dash(S0, S) :-
    ( sub_string(S0, 0, 1, _, "-") ->
        sub_string(S0, 1, _, 0, S1),
        string_trim(S1, S)
    ; string_trim(S0, S)
    ).

read_answers(In, Values) :-
    read_line_to_string(In, Line0),
    ( Line0 == end_of_file ->
        Values = []
    ; string_trim(Line0, Line),
      format(user_error, "[DBG] Read line: ~q~n", [Line]),
      ( is_done(Line) ->
          Values = []
      ; parse_answer_line_numeric(Line, Values1),
        read_answers(In, ValuesRest),
        append(Values1, ValuesRest, Values)
      )
    ).

is_done("done") :- !.
is_done(done)   :- !.

parse_answer_line_numeric(Line, Values) :-
    ( sub_string(Line, 0, _, _, "ans:") ->
        sub_string(Line, 4, _, 0, Rest),
        ( sub_string(Rest, EqPos, _, After, "=") ->
            sub_string(Rest, 0, EqPos, _, IdStr0),
            sub_string(Rest, _, After, 0, ValStr0),
            string_trim(IdStr0, IdStr),
            string_trim(ValStr0, ValStr),
            atom_string(Id, IdStr),
            ( catch(number_string(Num, ValStr), _, fail) ->
                Values = [value(Id, Num)]
            ;   Values = []
            )
        ; throw(bad_answer(Line))
        )
    ; throw(bad_answer(Line))
    ).

parse_fuzzy_rule_line(LineStr, Rule) :-
    string_codes(LineStr, Codes),
    phrase(fuzzy_rule(Rule), Codes).

fuzzy_rule(rule(Conn, Ante, Cons)) -->
    blanks, "-", blanks,
    ("If"; "if"), blanks,
    antecedent(Conn, Ante),
    blanks, ("then"; "Then"), blanks,
    consequent(Cons),
    blanks, ".", blanks.

antecedent(Conn, [L1, L2]) -->
    literal(L1),
    blanks,
    connector(Conn),
    blanks,
    literal(L2).
antecedent(and, [L]) -->
    literal(L).

connector(or)  --> ("or"; "OR").
connector(and) --> ("and"; "AND").

literal(Id/Pred) -->
    identifier(Id), blanks, ("is"; "IS"), blanks, identifier(Pred).

consequent(Id/Pred) --> literal(Id/Pred).

identifier(Atom) -->
    word_codes(Cs),
    { Cs \= [],
      string_codes(S, Cs),
      string_lower(S, SL),
      atom_string(Atom, SL)
    }.

word_codes([C|Cs]) -->
    [C],
    { code_type(C, alnum) ; C = 0'_; C = 0'- },
    !,
    word_codes_rest(Cs).
word_codes_rest([C|Cs]) -->
    [C],
    { code_type(C, alnum) ; C = 0'_; C = 0'- },
    !,
    word_codes_rest(Cs).
word_codes_rest([]) --> [].

clamp01(X, Y) :- Y is max(0.0, min(1.0, X)).

left_shoulder(A, B, X, Mu) :-
    ( X =< A -> Mu = 1.0
    ; X >= B -> Mu = 0.0
    ; Mu0 is (B - X) / (B - A), clamp01(Mu0, Mu)
    ).

right_shoulder(A, B, X, Mu) :-
    ( X =< A -> Mu = 0.0
    ; X >= B -> Mu = 1.0
    ; Mu0 is (X - A) / (B - A), clamp01(Mu0, Mu)
    ).

triangle(A, B, C, X, Mu) :-
    ( X =< A -> Mu = 0.0
    ; X >= C -> Mu = 0.0
    ; X =< B -> Mu0 is (X - A) / (B - A), clamp01(Mu0, Mu)
    ; Mu0 is (C - X) / (C - B), clamp01(Mu0, Mu)
    ).

mu(service, poor, X, Mu)      :- left_shoulder(0.0, 4.0, X, Mu).
mu(service, good, X, Mu)      :- triangle(2.0, 5.0, 8.0, X, Mu).
mu(service, excellent, X, Mu) :- right_shoulder(6.0, 10.0, X, Mu).

mu(food, rancid, X, Mu)       :- left_shoulder(0.0, 4.0, X, Mu).
mu(food, delicious, X, Mu)    :- right_shoulder(6.0, 10.0, X, Mu).

mu(tip, cheap, Y, Mu)         :- left_shoulder(0.0, 10.0, Y, Mu).
mu(tip, normal, Y, Mu)        :- triangle(5.0, 12.5, 20.0, Y, Mu).
mu(tip, generous, Y, Mu)      :- right_shoulder(15.0, 25.0, Y, Mu).

mu(cpu, slow, X, Mu)          :- left_shoulder(0.0, 4.0, X, Mu).
mu(cpu, normal, X, Mu)        :- triangle(2.0, 5.0, 8.0, X, Mu).
mu(cpu, fast, X, Mu)          :- right_shoulder(6.0, 10.0, X, Mu).

mu(ram, low, X, Mu)           :- left_shoulder(0.0, 4.0, X, Mu).
mu(ram, ok, X, Mu)            :- triangle(2.0, 5.0, 8.0, X, Mu).
mu(ram, high, X, Mu)          :- right_shoulder(6.0, 10.0, X, Mu).

mu(action, keep, Y, Mu)       :- left_shoulder(0.0, 10.0, Y, Mu).
mu(action, optimize, Y, Mu)   :- triangle(5.0, 12.5, 20.0, Y, Mu).
mu(action, upgrade, Y, Mu)    :- right_shoulder(15.0, 25.0, Y, Mu).

get_value(Id, Values, V) :- member(value(Id, V), Values).

lit_degree(Id/Pred, Values, Deg) :-
    get_value(Id, Values, X),
    mu(Id, Pred, X, Deg).

combine_degrees(or, Degs, Out)  :- max_list(Degs, Out).
combine_degrees(and, Degs, Out) :- min_list(Degs, Out).

rule_applicability(rule(Conn, Lits, _Cons), Values, Alpha) :-
    maplist({Values}/[Lit,Deg]>>lit_degree(Lit, Values, Deg), Lits, Degs),
    combine_degrees(Conn, Degs, Alpha).

output_domain(Domain) :- findall(Y, between(0,25,Y), Domain).

rule_contrib_at_y(rule(_Conn, _Lits, OutVar/OutPred), Alpha, Y, MuY) :-
    mu(OutVar, OutPred, Y, BaseMu),
    MuY is min(Alpha, BaseMu).

aggregate_output_curve(Rules, Values, Curve) :-
    output_domain(Domain),
    findall(alpha_rule(R,Alpha),
        ( member(R, Rules),
          rule_applicability(R, Values, Alpha)
        ),
        Alphas),
    findall(Y-MuAgg,
        ( member(Y, Domain),
          findall(MuY,
              ( member(alpha_rule(R,Alpha), Alphas),
                rule_contrib_at_y(R, Alpha, Y, MuY)
              ),
              Mus),
          ( Mus = [] -> MuAgg = 0.0 ; max_list(Mus, MuAgg) )
        ),
        Curve).

defuzzify_centroid(Curve, Crisp) :-
    foldl([_Y-Mu, S0, S1]>>(S1 is S0 + Mu), Curve, 0.0, SumMu),
    ( SumMu =:= 0.0 ->
        Crisp = 0.0
    ; foldl([Y-Mu, A0, A1]>>(A1 is A0 + Y*Mu), Curve, 0.0, SumYMu),
      Crisp is SumYMu / SumMu
    ).
