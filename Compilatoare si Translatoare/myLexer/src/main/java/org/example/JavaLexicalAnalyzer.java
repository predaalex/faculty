package org.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class JavaLexicalAnalyzer {

    private static final List<String> KEYWORDS = Arrays.asList(
            "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
            "continue", "default", "do", "double", "else", "enum", "extends", "false", "final", "finally",
            "float", "for", "if", "implements", "import", "instanceof", "int", "interface", "long", "native",
            "new", "null", "package", "private", "protected", "public", "return", "short", "static", "strictfp",
            "super", "switch", "synchronized", "this", "throw", "throws", "transient", "true", "try", "void",
            "volatile", "while"
    );

    private static final List<String> OPERATORS = Arrays.asList(
            "+", "-", "*", "/", "%", "&", "|", "^", "!", "~", "=", "<", ">", "?", ":",
            "++", "--", "*=", "/=", "%=", "%=", "|=", "^=", "=="
    );

    private static final List<String> DELIMITERS = Arrays.asList(
            ";", ",", ".", "(", ")", "{", "}", "[", "]"
    );

    public static void main(String[] args) {
        // pathul catre fisierul de input
        String fileName = "src/main/java/org/example/input.txt";
        String input = "";
        try {
            input = new String(Files.readAllBytes(Paths.get(fileName)));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // stergem codul comentat
        // bocuri comentate
        input = removeBlockComments(input);

        // single line comment
        input = input.replaceAll("//.*", "");


        List<Token> tokens = tokenize(input);

        for (Token token : tokens) {
            System.out.println(token);
        }
    }

    private static String removeBlockComments(String input) {
        Pattern pattern = Pattern.compile("/\\*.*?\\*/", Pattern.DOTALL);
        Matcher matcher = pattern.matcher(input);
        StringBuilder sb = new StringBuilder();
        while (matcher.find()) {
            matcher.appendReplacement(sb, "");
        }
        matcher.appendTail(sb);
        return sb.toString();
    }

    public static List<Token> tokenize(String input) {
        int line = 1;
        List<Token> tokens = new ArrayList<>();
        int i = 0;
        while (i < input.length()) {
            if (input.charAt(i) == '\n')
                line++;
            char c = input.charAt(i);
            if (Character.isWhitespace(c)) {
                i++;
            } else if (Character.isJavaIdentifierStart(c)) {
                StringBuilder sb = new StringBuilder();
                sb.append(c);
                i++;
                while (i < input.length() && Character.isJavaIdentifierPart(input.charAt(i))) {
                    sb.append(input.charAt(i));
                    i++;
                }
                String tokenValue = sb.toString();
                if (KEYWORDS.contains(tokenValue)) {
                    tokens.add(new Token(TokenType.KEYWORD, tokenValue, line));
                } else {
                    tokens.add(new Token(TokenType.IDENTIFIER, tokenValue, line));
                }
            } else if (c == '"') {
                StringBuilder sb = new StringBuilder();
                sb.append(c);
                i++;
                while (i < input.length() && input.charAt(i) != '"') {
                    sb.append(input.charAt(i));
                    i++;
                }
                sb.append(input.charAt(i));
                i++;
                tokens.add(new Token(TokenType.STRING, sb.toString(), line));
            } else if (OPERATORS.contains(String.valueOf(c))) {
                StringBuilder sb = new StringBuilder();
                sb.append(c);

                char nextChar = input.charAt(i + 1);
                sb.append(nextChar);

                // verific daca - este operator sau numar negativ
                if (String.valueOf(c).equals("-") && Character.isDigit(nextChar)) {
                    i += 2;
                    while (i < input.length() && Character.isDigit(input.charAt(i))) {
                        sb.append(input.charAt(i));
                        i++;
                    }
                    tokens.add(new Token(TokenType.NUMBER, sb.toString(), line));
                    continue;
                }



                // verific daca este operator simplu sau dublu
                if (OPERATORS.contains(sb.toString())) {
                    tokens.add(new Token(TokenType.OPERATOR, sb.toString(), line));
                    i++;
                } else {
                    tokens.add(new Token(TokenType.OPERATOR, String.valueOf(c), line));
                }

                i++;
            } else if (Character.isDigit(c)) {
                StringBuilder sb = new StringBuilder();
                sb.append(c);
                i++;
                boolean isFloat = false;
                while (i < input.length() && (Character.isDigit(input.charAt(i)) || input.charAt(i) == '.' )) {
                    if (input.charAt(i) == '.' && isFloat)
                        break;
                    if (input.charAt(i) == '.' && !isFloat)
                        isFloat = true;
                    sb.append(input.charAt(i));
                    i++;
                }
                if (isFloat) {
                    tokens.add(new Token(TokenType.FLOAT, sb.toString(), line));
                } else {
                    tokens.add(new Token(TokenType.NUMBER, sb.toString(), line));
                }
            } else if (DELIMITERS.contains(String.valueOf(c))) {
                StringBuilder sb = new StringBuilder();
                sb.append(c);

                char nextChar = input.charAt(i + 1);

                // verific daca . este delimitator sau numar float
                if (String.valueOf(c).equals(".") && Character.isDigit(nextChar)) {
                    sb.append(nextChar);
                    i += 2;
                    while (i < input.length() && Character.isDigit(input.charAt(i))) {
                        sb.append(input.charAt(i));
                        i++;
                    }
                    tokens.add(new Token(TokenType.FLOAT, sb.toString(), line));
                    continue;
                }

                tokens.add(new Token(TokenType.DELIMITER, sb.toString(), line));
                i++;
            } else {
                // unrecognized character
                tokens.add(new Token(TokenType.ERROR, String.valueOf(c), line));
                i++;
            }
        }
        return tokens;
    }

    public static class Token {
        private TokenType type;
        private String value;
        private int line;

        public Token(TokenType type, String value, int line) {
            this.type = type;
            this.value = value;
            this.line = line;
        }

        public TokenType getType() {
            return type;
        }

        public String getValue() {
            return value;
        }

        public int getLine() {
            return line;
        }

        public void setLine(int line) {
            this.line = line;
        }

        public String toString() {
            return String.format("'%s', %s; %d; linia %d", value, type, value.length(), line);
        }
    }

    public enum TokenType {
        KEYWORD,
        IDENTIFIER,
        NUMBER,
        FLOAT,
        STRING,
        OPERATOR,
        DELIMITER,
        ERROR
    }
}
