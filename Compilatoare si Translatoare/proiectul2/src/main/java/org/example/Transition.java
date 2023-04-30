package org.example;

public class Transition {
    public String value;
    public int target;

    public Transition(String value, int target) {
        this.value = value;
        this.target = target;
    }

    @Override
    public String toString() {
        return "Transition{" +
                "value='" + value + '\'' +
                ", target=" + target +
                '}';
    }
}
