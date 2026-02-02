package p2_gui.ex2;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;

public class DiagnosisAlgorithm {

    private final String pathExecSwipl = "C:\\Program Files\\swipl\\bin\\swipl.exe";
    private final String pathAlgorithm = "C:\\Users\\allex\\Desktop\\git_repos\\faculty\\KRR\\project2\\ExempluInterfataPrologSwi\\engine2_no_comments.pl";

    public static final String PORT = "5007";

    private Process processSwipl;

    public String runWithAnswers(File selectedFile, List<String> answerLines) throws IOException {
        int port = Integer.parseInt(PORT);

        try (ServerSocket serverSocket = new ServerSocket(port)) {

            String commandString = getCommandString(selectedFile.getPath());
            System.out.println("Running Prolog command: " + commandString);

            processSwipl = Runtime.getRuntime().exec(commandString);
            consumeStreamAsync(processSwipl.getErrorStream(), "SWI-ERR");

            try (Socket socket = serverSocket.accept();
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {

                // Send answers to Prolog
                for (String line : answerLines) {
                    System.out.println("JAVA:" + line);
                    out.println(line);
                }
                out.println("done"); // end marker

                // Read result
                String line;
                String finalResult = null;

                while ((line = in.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;

                    String lower = line.toLowerCase();
                    System.out.println("Prolog: " + line);

//                    if (lower.equals("entailed") || lower.equals("not_entailed")) {
                    finalResult = lower;
                    break;
//                    }
                }

                if (finalResult == null) {
                    throw new IOException("No final response from Prolog");
                }

                return finalResult;

            } finally {
                if (processSwipl != null) processSwipl.destroy();
            }
        }
    }

    private String getCommandString(String filePath) {
        return "\"" + pathExecSwipl + "\""
                + " -s " + "\"" + pathAlgorithm + "\""
                + " -- " + PORT
                + " " + "\"" + filePath + "\"";
    }

    private void consumeStreamAsync(InputStream is, String name) {
        new Thread(() -> {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
                String line;
                while ((line = br.readLine()) != null) {
                    System.out.println(name + ": " + line);
                }
            } catch (IOException e) {
                // ignore
            }
        }, name + "-reader").start();
    }
}