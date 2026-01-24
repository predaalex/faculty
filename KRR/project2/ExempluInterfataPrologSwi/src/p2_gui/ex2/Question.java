package p2_gui.ex2;

public class Question {
    public enum Type { NUMBER, STRING, BOOLEAN }

    private final String id;          // machine key, e.g., days_sick
    private final String labelText;   // shown in UI
    private final Type type;

    public Question(String id, String labelText, Type type) {
        this.id = id;
        this.labelText = labelText;
        this.type = type;
    }

    public String getId() { return id; }
    public String getLabelText() { return labelText; }
    public Type getType() { return type; }
}
