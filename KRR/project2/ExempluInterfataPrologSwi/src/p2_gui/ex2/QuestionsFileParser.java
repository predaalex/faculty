package p2_gui.ex2;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class QuestionsFileParser {

    private static final Pattern QLINE = Pattern.compile("^-[\\[]([^\\]]+)[\\]]\\s*(.+)$");

    public static List<Question> parse(File file) throws IOException {
        List<Question> questions = new ArrayList<>();
        boolean inQuestions = false;

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {

            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                if (line.equalsIgnoreCase("Questions:")) {
                    inQuestions = true;
                    continue;
                }
                if (line.equalsIgnoreCase("The goal:")) {
                    inQuestions = false;
                    continue;
                }
                if (!inQuestions) continue;

                Matcher m = QLINE.matcher(line);
                if (!m.matches()) {
                    throw new IllegalArgumentException("Bad question line (missing -[id]): " + line);
                }

                String id = m.group(1).trim();
                String rest = m.group(2).trim();

                Question.Type type = inferType(rest);
                String label = stripTypeSuffix(rest);

                questions.add(new Question(id, label, type));
            }
        }

        return questions;
    }

    private static Question.Type inferType(String line) {
        String lower = line.toLowerCase();
        if (lower.contains("(number)")) return Question.Type.NUMBER;
        if (lower.contains("(boolean)")) return Question.Type.BOOLEAN;
        if (lower.contains("(string)")) return Question.Type.STRING;
        return Question.Type.STRING;
    }

    private static String stripTypeSuffix(String line) {
        return line
                .replace("(number)", "").replace("(NUMBER)", "")
                .replace("(boolean)", "").replace("(BOOLEAN)", "")
                .replace("(string)", "").replace("(STRING)", "")
                .trim();
    }
}
