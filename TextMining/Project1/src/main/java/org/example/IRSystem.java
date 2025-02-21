package org.example;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.ro.RomanianAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.tika.Tika;
import org.apache.tika.exception.TikaException;
import org.bouncycastle.util.encoders.UTF8;


public class IRSystem {
    public static void main(String[] args) throws Exception {
        System.setProperty("file.encoding","UTF-8");
        System.out.println(System.getProperty("file.encoding"));

        System.out.println(args[2]);
        System.out.println(Arrays.toString(args));

        if (args.length < 2) {
            System.err.println("Usage: -index -directory <path to docs> or -search -query <keyword>");
            return;
        }

        String mode = args[0];

        if (mode.equals("-index")) {
            if (args.length < 3) {
                System.err.println("Missing directory argument for indexing.");
                return;
            }
            String folderPath = args[2];
            indexDocuments(folderPath);
        } else if (mode.equals("-search")) {
            if (args.length < 3) {
                System.err.println("Missing query argument for searching.");
                return;
            }
            // Citim propoziția întreagă din linia de comandă folosind toate argumentele după "-query"
            String querystr = String.join(" ", args).split("-query ")[1];
            System.out.println(searchDocuments(querystr));
        } else {
            System.err.println("Unknown mode. Use -index or -search.");
        }
    }

    // Metoda pentru indexarea documentelor
    public static void indexDocuments(String folderPath) throws IOException, TikaException {
        RomanianAnalyzer analyzer = new RomanianAnalyzer();
        Directory index = FSDirectory.open(Path.of("index"));
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter writer = new IndexWriter(index, config);

        List<Document> documents = readDocumentsFromFolder(folderPath);
        System.out.println("Found " + documents.size() + " documents.");

        for (Document doc : documents) {
            writer.addDocument(doc);
        }
        writer.close();
        System.out.println("Indexing completed.");
    }

    // Metoda pentru căutarea documentelor
    public static ArrayList<String> searchDocuments(String querystr) throws IOException, ParseException {
        RomanianAnalyzer analyzer = new RomanianAnalyzer();
        Directory index = FSDirectory.open(Path.of("index"));
        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);
//        System.out.println(querystr);
        // Preprocesăm interogarea
        String preprocessedQuery = preprocessText(querystr);
        System.out.println("Original Query:" + querystr);
        System.out.println("Processed Query: " + preprocessedQuery); // DEBUG
        String[] terms = preprocessedQuery.split("\\s+");
        StringBuilder booleanQuery = new StringBuilder();
        for (String term : terms) {
            if (!term.isEmpty()) {
                booleanQuery.append(term).append(" OR ");
            }
        }

        // Eliminăm ultimul "AND"
        if (booleanQuery.length() > 0) {
            booleanQuery.setLength(booleanQuery.length() - 4);
        }

//        System.out.println(booleanQuery);
        Query q = new QueryParser("content", analyzer).parse(booleanQuery.toString());


        // Căutăm și afișăm primele 5 rezultate
        TopDocs docs = searcher.search(q, 3);
        ScoreDoc[] hits = docs.scoreDocs;
        ArrayList<String> result = new ArrayList<>();

        for (ScoreDoc hit : hits) {
            Document d = searcher.doc(hit.doc);
            result.add(d.get("fileName"));
        }

        reader.close();

        return result;
    }

    private static String stemmingText(String query) {
        RomanianAnalyzer analyzer = new RomanianAnalyzer();

        try {
            // Creăm un TokenStream pentru interogare
            TokenStream tokenStream = analyzer.tokenStream("content", query);
            CharTermAttribute charTermAttribute = tokenStream.addAttribute(CharTermAttribute.class);
            tokenStream.reset();
            StringBuilder processedQuery = new StringBuilder();

            // Aplicăm stemming pe fiecare cuvânt, dar păstrăm structura originală
            String[] originalWords = query.split("\\s+");
            int wordIndex = 0;

            while (tokenStream.incrementToken()) {
                // Preluăm cuvântul stemmat și îl înlocuim în propoziția originală
                if (wordIndex < originalWords.length) {
                    processedQuery.append(charTermAttribute.toString()).append(" ");
                    wordIndex++;
                }
            }
            tokenStream.end();
            tokenStream.close();

            return processedQuery.toString().trim();
        } catch (IOException e) {
            System.err.println("Error preprocessing query: " + e.getMessage());
            return query;
        }
    }


    // Metoda pentru eliminarea diacriticelor
    private static String removeDiacritics(String text) {
        String[][] diacriticsMap = {
                {"ă", "a"}, {"â", "a"}, {"î", "i"}, {"ș", "s"}, {"ț", "t"},
                {"Ă", "A"}, {"Â", "A"}, {"Î", "I"}, {"Ș", "S"}, {"Ț", "T"}
        };

        // Înlocuim fiecare literă cu diacritice cu echivalentul său fără diacritice
        for (String[] mapping : diacriticsMap) {
            text = text.replace(mapping[0], mapping[1]);
        }

        return text;
    }

    // Metoda pentru citirea documentelor dintr-un folder
    private static List<Document> readDocumentsFromFolder(String folderPath) {
        List<Document> documents = new ArrayList<>();
        File folder = new File(folderPath);
        Tika tika = new Tika();

        if (folder.exists() && folder.isDirectory()) {
            for (File file : folder.listFiles()) {
                if (file.isFile() && isSupportedFileType(file.getName())) {
                    try (InputStream inputStream = new FileInputStream(file)) {
                        String content = tika.parseToString(inputStream);

                        System.out.println("File Name: " + file.getName() + "\nContent: " + content + "\n");
                        String preprocessedContent = preprocessText(content);
                        Document doc = new Document();
                        System.out.println(file.getAbsolutePath());
                        doc.add(new TextField("fileName", file.getName(), Field.Store.YES));
                        doc.add(new TextField("content", preprocessedContent, Field.Store.YES));
                        documents.add(doc);
                    } catch (IOException | TikaException e) {
                        System.err.println("Error reading file: " + file.getName());
                    }
                }
            }
        } else {
            System.err.println("Folder " + folderPath + " does not exist.");
        }

        return documents;
    }

    // Metoda pentru preprocesarea textului (eliminarea diacriticelor)
    private static String preprocessText(String text) {
        text = removeDiacritics(text);
        text = stemmingText(text);

        return text;
    }

    // Verifică dacă tipul de fișier este suportat
    private static boolean isSupportedFileType(String fileName) {
        return fileName.endsWith(".txt") || fileName.endsWith(".pdf") || fileName.endsWith(".docx");
    }
}