package p1_gui;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.List;

public class ResolutionWindow extends JFrame {

    private JComboBox<String> fileComboBoxResolution;
    private JComboBox<String> fileComboBoxSATSolver;
    private JButton runButtonResolution;
    private JButton runButtonSATSolver;
    private JTextField resultFieldResolution;
    private JTextField resultFieldSATSolver;
    private File resourcesDirResolution;
    private File resourcesDirSATSolver;

    public ResolutionWindow() {
        initComponents();
        setupLayout();
        setupListeners();

        setTitle("Project 1");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null); // center
    }

    private void initComponents() {
        resourcesDirResolution = new File("resources/s1");
        resourcesDirSATSolver = new File("resources/s2");

        if (!resourcesDirResolution.exists() || !resourcesDirResolution.isDirectory()) {
            JOptionPane.showMessageDialog(
                    this,
                    "Could not find 'resources' directory next to project root.",
                    "Warning",
                    JOptionPane.WARNING_MESSAGE
            );
            fileComboBoxResolution = new JComboBox<>(new String[]{});
        } else {
            File[] files = resourcesDirResolution.listFiles(File::isFile);
            String[] fileNames;
            if (files == null || files.length == 0) {
                fileNames = new String[]{};
            } else {
                fileNames = new String[files.length];
                for (int i = 0; i < files.length; i++) {
                    fileNames[i] = files[i].getName();
                }
            }
            fileComboBoxResolution = new JComboBox<>(fileNames);
        }

        if (!resourcesDirSATSolver.exists() || !resourcesDirSATSolver.isDirectory()) {
            JOptionPane.showMessageDialog(
                    this,
                    "Could not find 'resources' directory next to project root.",
                    "Warning",
                    JOptionPane.WARNING_MESSAGE
            );
            fileComboBoxSATSolver = new JComboBox<>(new String[]{});
        } else {
            File[] files = resourcesDirSATSolver.listFiles(File::isFile);
            String[] fileNames;
            if (files == null || files.length == 0) {
                fileNames = new String[]{};
            } else {
                fileNames = new String[files.length];
                for (int i = 0; i < files.length; i++) {
                    fileNames[i] = files[i].getName();
                }
            }
            fileComboBoxSATSolver = new JComboBox<>(fileNames);
        }

        runButtonResolution = new JButton("Run Resolution");
        runButtonSATSolver = new JButton("Run SAT Solver");


        resultFieldResolution = new JTextField(20);
        resultFieldResolution.setEditable(false);

        resultFieldSATSolver = new JTextField(20);
        resultFieldSATSolver.setEditable(false);
    }

    private void setupLayout() {
        JPanel main = new JPanel();
        main.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        main.setLayout(new BoxLayout(main, BoxLayout.Y_AXIS));

        JPanel row1 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row1.add(new JLabel("Select Kb+Q file:"));
        row1.add(fileComboBoxResolution);

        JPanel row2 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row2.add(runButtonResolution);

        JPanel row3 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row3.add(new JLabel("Result:"));
        row3.add(resultFieldResolution);

        main.add(row1);
        main.add(row2);
        main.add(row3);

        main.add(new JSeparator(JSeparator.HORIZONTAL));

        JPanel row4 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row4.add(new JLabel("Select Set of Clauses file:"));
        row4.add(fileComboBoxSATSolver);

        JPanel row5 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row5.add(runButtonSATSolver);

        JPanel row6 = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row6.add(new JLabel("Result:"));
        row6.add(resultFieldSATSolver);

        main.add(row4);
        main.add(row5);
        main.add(row6);

        setContentPane(main);
    }

    private void setupListeners() {
        runButtonResolution.addActionListener(e -> runResolution());
        runButtonSATSolver.addActionListener(e -> runSATSolver());
    }

    private void runResolution() {
        String selectedFileName = (String) fileComboBoxResolution.getSelectedItem();
        File file = new File(resourcesDirResolution, selectedFileName);
        FolAlgorithm folAlgorithm = new FolAlgorithm();
        boolean entailed;

        try {
            entailed = folAlgorithm.isEntailed(file);
        } catch (Exception ex) {
            ex.printStackTrace();
            resultFieldResolution.setText("Error: " + ex.getMessage());
            return;
        }

        if (entailed) {
            resultFieldResolution.setText("is entailed");
        } else {
            resultFieldResolution.setText("is not entailed");
        }
    }

    private void runSATSolver() {
        String selectedFileName = (String) fileComboBoxSATSolver.getSelectedItem();
        File file = new File(resourcesDirSATSolver, selectedFileName);
        SatAlgorithm SatAlgorithm = new SatAlgorithm();
        List<String> result;

        try {
            result = SatAlgorithm.isSatisfiable(file);
        } catch (Exception ex) {
            ex.printStackTrace();
            resultFieldSATSolver.setText("Error: " + ex.getMessage());
            return;
        }

        if (result.isEmpty()) {
            resultFieldSATSolver.setText("NO");
        } else {
            resultFieldSATSolver.setText(result.toString());
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new ResolutionWindow().setVisible(true));
    }
}
