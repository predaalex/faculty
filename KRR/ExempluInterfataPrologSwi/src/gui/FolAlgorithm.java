package gui;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;

public class FolAlgorithm {
    // adjust this path to your SWI Prolog installation
    private final String pathExecSwipl = "C:\\Program Files\\swipl\\bin\\swipl.exe";
    private final String pathAlgorithm = "C:\\Users\\allex\\Desktop\\git_repos\\faculty\\KRR\\ExempluInterfataPrologSwi\\resolution.pl";

    // same port that Prolog will use to connect back
    public static final String PORT = "5004";

    private Process processSwipl;

    public boolean isEntailed(File file) throws IOException {
        int port = Integer.parseInt(PORT);

        // 1. Open a server socket so Prolog can connect back
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            String commandString = getCommandString(file.getPath());
            System.out.println("Running Prolog command: " + commandString);

            // 2. Start SWI Prolog
            processSwipl = Runtime.getRuntime().exec(commandString);

            // (Optional) eat stderr so Prolog can't block on it
            consumeStreamAsync(processSwipl.getErrorStream(), "SWI-ERR");

            // 3. Wait for Prolog to connect
            try (Socket socket = serverSocket.accept();
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {

                // If you want to send something initially to Prolog, use 'out.println(...)' here

                String line;
                String finalResult = null;

                while ((line = in.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) {
                        continue;
                    }

                    String lower = line.toLowerCase();

                    // Check for final answer from Prolog
                    if (lower.equals("entailed") || lower.equals("not_entailed")) {
                        System.out.println("Prolog final response: " + line);
                        finalResult = lower;
                        break;
                    } else {
                        // This is a trace line (e.g. starts with "TRACE: ...")
                        System.out.println("Prolog trace: " + line);
                        // If you have a JTextArea in your UI, you could also do:
                        // textAreaDebug.append(line + "\n");
                    }
                }

                if (finalResult == null) {
                    throw new IOException("No final entailed/not_entailed response from Prolog");
                }

                // Interpret final result
                if (finalResult.equals("entailed")) {
                    return true;
                } else { // "not_entailed"
                    return false;
                }

            } finally {
                // Clean up process
                if (processSwipl != null) {
                    processSwipl.destroy();
                }
            }

        }
    }

    // Build the swipl command
    // We use:
    //   swipl.exe -s "path\to\file.pl" -- 5004
    // and in Prolog we define `:- initialization(main, main).` so main/1 gets ["5004"].
    public String getCommandString(String filePath) {
        return "\"" + pathExecSwipl + "\""
                + " -s " + "\"" + pathAlgorithm + "\""
                + " -- " + PORT
                + " " + filePath;
    }

    // Helper to dump stderr asynchronously (optional, useful for debugging)
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
