# Document Search with Romanian Language Support

## Overview

This Java program is a simple document search engine built using Apache Lucene and Apache Tika, with enhancements for the Romanian language. It supports indexing documents (`.txt`, `.pdf`, `.docx`) and searching for keywords or phrases, taking into account Romanian diacritics and stemming.

## Features Added

1. **Romanian Language Preprocessing:**
    - Integrated `RomanianAnalyzer` from Apache Lucene to handle:
        - **Stemming**: Normalizes Romanian words (e.g., "mamei" → "mam").
        - **Stop Words Removal**: Automatically filters out common Romanian stop words (e.g., "și", "sau").

2. **Diacritics Handling:**
    - Added a method to **remove diacritics** from both the indexed content and user queries.
    - Ensures consistent search results whether the input contains diacritics or not (e.g., "piață" and "piata" are treated equally).

3. **Phrase Query Support:**
    - Enabled searching for **exact phrases** by detecting if the query contains multiple words.
    - Uses `QueryParser` to construct **phrase queries** when needed (e.g., `"mama merge la piață"` is treated as an exact phrase).

4. **Improved Command-Line Argument Handling:**
    - Added support for reading the entire query string (including spaces) from the command line.
    - Supports searches with both single words and full sentences.


# How to Use

Follow these steps to build, index documents, and search using the command line:

1. **Index Documents:**
   - Run this command to index all documents from the specified folder:
     ```shell
     java -Dfile.encoding=UTF-8 -jar target/docsearch-1.0-SNAPSHOT.jar -index -directory <path to docs>
     ```

2. **Search for a Keyword or Phrase:**
   - To search for a single keyword:
     ```shell
     java -Dfile.encoding=UTF-8 -jar target/docsearch-1.0-SNAPSHOT.jar -search -query <keyword>
     ```

   - To search for an entire phrase or sentence:
     ```shell
     java -Dfile.encoding=UTF-8 -jar target/docsearch-1.0-SNAPSHOT.jar -search -query "<your phrase here>"
     ```

### Notes:
- Use `-Dfile.encoding=UTF-8` to ensure correct handling of Romanian diacritics.
- The `<path to docs>` should be replaced with the directory containing your documents.
- The `<keyword>` or `<your phrase here>` should be replaced with the term or sentence you want to search for. 
- Make sure the diacritics from string query search are utf-8 format.