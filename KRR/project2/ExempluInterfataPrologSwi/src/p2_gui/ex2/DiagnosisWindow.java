package p2_gui.ex2;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class DiagnosisWindow extends JFrame {

    private JComboBox<String> fileCombo;
    private File resourcesDir;

    private DynamicAnswerPanel dynamicPanel;
    private JButton runButton;
    private JTextField resultField;

    private List<Question> currentQuestions = new ArrayList<>();

    public DiagnosisWindow() {
        initComponents();
        setupLayout();
        setupListeners();

        setTitle("Project 2");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }

    private void initComponents() {
        resourcesDir = new File("resources/lab6");

        String[] fileNames = listFiles(resourcesDir);
        fileCombo = new JComboBox<>(fileNames);

        dynamicPanel = new DynamicAnswerPanel();

        runButton = new JButton("Run (Prolog)");
        resultField = new JTextField(30);
        resultField.setEditable(false);

        // Load first file automatically if exists
        if (fileNames.length > 0) {
            loadQuestionsForSelectedFile();
        }
    }

    private void setupLayout() {
        JPanel main = new JPanel();
        main.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        main.setLayout(new BoxLayout(main, BoxLayout.Y_AXIS));

        JLabel exerciseNameLabel1 = new JLabel("Exercise 2");

        JPanel row1 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row1.add(new JLabel("Select file:"));
        row1.add(fileCombo);

        JPanel row2 = new JPanel(new BorderLayout());
        row2.add(new JScrollPane(dynamicPanel), BorderLayout.CENTER);

        JPanel row3 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row3.add(runButton);

        JPanel row4 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row4.add(new JLabel("Result:"));
        row4.add(resultField);

        main.add(exerciseNameLabel1);
        main.add(row1);
        main.add(row2);
        main.add(row3);
        main.add(row4);

        main.add(new JSeparator(JSeparator.HORIZONTAL));

        setContentPane(main);
    }

    private void setupListeners() {
        fileCombo.addActionListener(e -> loadQuestionsForSelectedFile());
        runButton.addActionListener(e -> runDiagnosis());
    }

    private void loadQuestionsForSelectedFile() {
        resultField.setText("");

        if (!resourcesDir.exists() || !resourcesDir.isDirectory()) {
            JOptionPane.showMessageDialog(this,
                    "Could not find directory: resources/lab5",
                    "Warning",
                    JOptionPane.WARNING_MESSAGE);
            return;
        }

        String selected = (String) fileCombo.getSelectedItem();
        if (selected == null) return;

        File file = new File(resourcesDir, selected);

        try {
            currentQuestions = QuestionsFileParser.parse(file);
            dynamicPanel.setQuestions(currentQuestions);
        } catch (Exception ex) {
            ex.printStackTrace();
            resultField.setText("Error reading questions file");
        }
    }

    private void runDiagnosis() {
        resultField.setText("");

        String selected = (String) fileCombo.getSelectedItem();
        if (selected == null) {
            resultField.setText("Error: no file selected");
            return;
        }

        File file = new File(resourcesDir, selected);

        Map<Question, Object> answers;
        try {
            answers = dynamicPanel.readAnswersOrThrow();
        } catch (Exception ex) {
            resultField.setText("Error: " + ex.getMessage());
            return;
        }

        // Convert answers to lines to send to Prolog (type:value)
        List<String> answerLines = new ArrayList<>();
        for (Question q : currentQuestions) {
            Object v = answers.get(q);

            String val;
            switch (q.getType()) {
                case NUMBER:
                    val = String.valueOf(v);
                    break;
                case BOOLEAN:
                    val = ((Boolean) v) ? "yes" : "no";
                    break;
                default:
                    val = (v == null ? "" : v.toString());
            }

            answerLines.add("ans:" + q.getId() + "=" + val);
        }

        DiagnosisAlgorithm alg = new DiagnosisAlgorithm();
        try {
            String res = alg.runWithAnswers(file, answerLines);
            res = res.split(":")[1];
            resultField.setText(res);
        } catch (Exception ex) {
            ex.printStackTrace();
            resultField.setText("Error: " + ex.getMessage());
        }
    }

    private String[] listFiles(File dir) {
        if (!dir.exists() || !dir.isDirectory()) return new String[]{};

        File[] files = dir.listFiles(File::isFile);
        if (files == null || files.length == 0) return new String[]{};

        String[] names = new String[files.length];
        for (int i = 0; i < files.length; i++) names[i] = files[i].getName();
        return names;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new DiagnosisWindow().setVisible(true));
    }
}
