package org.example;

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
import py4j.GatewayServer;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.example.IRSystem.indexDocuments;
import static org.example.IRSystem.searchDocuments;


public class IRSystemAPI {
    // Expose the indexing method
    public void index(String folderPath) throws Exception {
        System.out.println("New index generated!");
        IRSystem.indexDocuments(folderPath);
    }

    // Expose the search method
    public ArrayList<String> search(String querystr) throws Exception {
        System.out.println("processed query: " + querystr );
        return searchDocuments(querystr);
    }

    public static void main(String[] args) {
        IRSystemAPI app = new IRSystemAPI();
        GatewayServer server = new GatewayServer(app);
        server.start();
        System.out.println("Gateway Server Started. Ready to accept requests...");
    }
}