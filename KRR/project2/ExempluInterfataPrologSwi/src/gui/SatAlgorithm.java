package gui;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class SatAlgorithm {
    public final String pathExecSwipl = "C:\\Program Files\\swipl\\bin\\swipl.exe";
    public final String pathAlgorithm = "C:\\Users\\allex\\Desktop\\git_repos\\faculty\\KRR\\project2\\ExempluInterfataPrologSwi\\SAT_Solver.pl";

    public static final String PORT = "5005";

    private Process processSwipl;

    public List<String> isSatisfiable(File file) throws IOException {
        int port = Integer.parseInt(PORT);

        // 1. Open server socket
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            String commandString = getCommandString(file.getPath());
            System.out.println("Running SAT Prolog command: " + commandString);

            // 2. Launch Prolog
            processSwipl = Runtime.getRuntime().exec(commandString);

            // Optional: stream stderr for debugging
            consumeStreamAsync(processSwipl.getErrorStream(), "SWI-ERR");

            // 3. Wait for Prolog to connect and reply
            try (Socket socket = serverSocket.accept();
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {

                String line;
                List<String> result = new ArrayList<>();

                while ((line = in.readLine()) != null) {
                    line = line.trim();
                    System.out.println("Prolog response: " + line);
                    if (line.isEmpty()) continue;

                    if (line.equalsIgnoreCase("yes")) {
                        continue;
                    } else if (line.equalsIgnoreCase("done")) {
                        break;
                    } else if (line.contains("/")) {
                        result.add(line);
                    } else {
//                        System.out.println("Unrecognized: " + line);
                    }
                }

                return result;
            } finally {
                if (processSwipl != null)
                    processSwipl.destroy();
            }
        }
    }
    public String getCommandString(String filePath) {
        return "\"" + pathExecSwipl + "\""
                + " -s " + "\"" + pathAlgorithm + "\""
                + " -- " + PORT
                + " " + filePath;
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
