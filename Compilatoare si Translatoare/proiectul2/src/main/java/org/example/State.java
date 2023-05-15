package org.example;

import java.util.ArrayList;

public class State {
    public int index;
    public boolean isFinal;
    public ArrayList<Transition> transitions;

    public State(int index, boolean isFinal, ArrayList<Transition> transitions) {
        this.index = index;
        this.isFinal = isFinal;
        this.transitions = transitions;
    }

    @Override
    public String toString() {
        return "State{" +
                "index=" + index +
                ", isFinal=" + isFinal +
                ", transitions=" + transitions +
                '}';
    }
}
