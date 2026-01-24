package p2_gui.ex1;

import javax.swing.*;
import java.awt.*;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class DynamicAnswerPanel extends JPanel {

    // Keep insertion order: question -> component
    private final Map<Question, JComponent> inputs = new LinkedHashMap<>();

    public DynamicAnswerPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
    }

    public void setQuestions(List<Question> questions) {
        removeAll();
        inputs.clear();

        for (Question q : questions) {
            add(buildRow(q));
        }

        revalidate();
        repaint();
    }

    public Map<Question, Object> readAnswersOrThrow() {
        Map<Question, Object> answers = new LinkedHashMap<>();

        for (Map.Entry<Question, JComponent> e : inputs.entrySet()) {
            Question q = e.getKey();
            JComponent comp = e.getValue();

            Object value;
            switch (q.getType()) {
                case NUMBER:
                    JTextField tf = (JTextField) comp;
                    String txt = tf.getText().trim();
                    if (txt.isEmpty()) throw new IllegalArgumentException("Missing number for: " + q.getLabelText());
                    try {
                        value = (Object) Double.parseDouble(txt);
                    } catch (Exception ex) {
                        throw new IllegalArgumentException("Invalid number for: " + q.getLabelText());
                    }
                    break;

                case STRING:
                    JTextField ts = (JTextField) comp;
                    value = ts.getText(); // empty string allowed
                    break;

                case BOOLEAN:
                    JPanel radioPanel = (JPanel) comp;
                    // We store the ButtonGroup in client property
                    ButtonGroup group = (ButtonGroup) radioPanel.getClientProperty("group");
                    if (group == null || group.getSelection() == null)
                        throw new IllegalArgumentException("Missing boolean for: " + q.getLabelText());
                    String sel = group.getSelection().getActionCommand();
                    value = (Object) sel.equalsIgnoreCase("yes");
                    break;

                default:
                    throw new IllegalStateException("Unknown question type");
            }

            answers.put(q, value);
        }

        return answers;
    }

    private JPanel buildRow(Question q) {
        JPanel row = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row.add(new JLabel(q.getLabelText()));

        JComponent input;

        if (q.getType() == Question.Type.BOOLEAN) {
            JRadioButton yes = new JRadioButton("Yes");
            yes.setActionCommand("yes");
            JRadioButton no = new JRadioButton("No");
            no.setActionCommand("no");

            ButtonGroup g = new ButtonGroup();
            g.add(yes);
            g.add(no);

            JPanel p = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
            p.putClientProperty("group", g);
            p.add(yes);
            p.add(no);

            input = p;
        } else {
            input = new JTextField(18);
        }

        inputs.put(q, input);
        row.add(input);

        return row;
    }
}
