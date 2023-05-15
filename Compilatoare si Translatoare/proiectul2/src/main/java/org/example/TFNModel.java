package org.example;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Scanner;

public class TFNModel {
    public ArrayList<State> stateArrayList;

    public TFNModel() {
        stateArrayList = new ArrayList<>();
    }

    @Override
    public String toString() {
        return "TFNModel{" +
                "stateArrayList=" + stateArrayList +
                '}';
    }

    public static void main(String[] args) {
        // functia de initializare tfn
        TFNModel tfn = tfnInit();

        // citim sirurile si cautam pathurile valide
        try (Scanner scanner = new Scanner(new File("src\\main\\resources\\input.txt"))) {
            while (scanner.hasNextLine()) {
                String inputLine = scanner.nextLine();
                System.out.println(inputLine + "\nAcest sir contine urmatoarele pathuri valide:\n" +
                        tfn.pathFinder(inputLine) + "\n");
            }
        } catch (FileNotFoundException fnfe) {
            fnfe.printStackTrace();
        }

    }

    private ArrayList<String> pathFinder(String inputLine) {
        ArrayList<String> paths = new ArrayList<>();
        dfsPath(this.stateArrayList.get(0), inputLine, paths, new StringBuilder());

        return paths;
    }

    private void dfsPath(State state, String inputLine, ArrayList<String> paths, StringBuilder path) {
        // daca starea este finala, adaugam path ul si revenim in dfs
        if (state.isFinal) {
            paths.add(path.toString());
            // daca starea nu este finala, cautam in tranzitiile starii daca putem trece mai departe cu primul caracter din alfabet
        } else {
            for (Transition transition : state.transitions) {
                if (transition.value == null) {
                    dfsPath(
                            Objects.requireNonNull(getStateWithIndex(transition.target)),
                            inputLine,
                            paths,
                            new StringBuilder(path).append("Q").append(state.index).append(" ")
                    );
                } else if (!inputLine.isEmpty() && transition.value.equals(String.valueOf(inputLine.charAt(0)))) {
                    dfsPath(
                            Objects.requireNonNull(getStateWithIndex(transition.target)),
                            inputLine.substring(1),
                            paths,
                            new StringBuilder(path).append("Q").append(state.index).append(" ")
                    );
                }
            }

        }

    }

    private State getStateWithIndex(int target) {
        for (State state : this.stateArrayList)
            if (state.index == target)
                return state;
        return null;
    }

    private static TFNModel tfnInit() {
        TFNModel tfn = new TFNModel();
        // 0 state
        ArrayList<Transition> transitions = new ArrayList<>();
        transitions.add(new Transition("a", 1));

        tfn.stateArrayList.add(new State(0, false, transitions));

        // 1 state
        transitions = new ArrayList<>();
        transitions.add(new Transition("b", 1));
        transitions.add(new Transition(null, 2));
        transitions.add(new Transition("a", 3));

        tfn.stateArrayList.add(new State(1, false, transitions));

        // 2 state
        transitions = new ArrayList<>();
        transitions.add(new Transition("b", 3));
        transitions.add(new Transition("c", 4));

        tfn.stateArrayList.add(new State(2, false, transitions));

        // 3 state
        transitions = new ArrayList<>();
        transitions.add(new Transition("c", 4));

        tfn.stateArrayList.add(new State(3, true, transitions));

        // 4 state
        transitions = new ArrayList<>();
        transitions.add(new Transition("a", 2));

        tfn.stateArrayList.add(new State(4, false, transitions));
        return tfn;
    }
}