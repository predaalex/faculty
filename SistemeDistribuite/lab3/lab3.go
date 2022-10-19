package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/mail"
	"os"
	"strconv"
	// "regexp"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}
func main() {
	path := "content/test.txt"

	content, err := os.ReadFile(path)

	if errors.Is(err, os.ErrNotExist) {
		log.Fatal("File does not exist!")
	} else if err != nil {
		log.Fatal("An error has occured!")
	}

	fmt.Println(string(content))


	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		log.Fatal("File does not exist!")
	} else if err != nil {
		log.Fatal("An error has occured!")
	}

	fileScanner := bufio.NewScanner(file)
	fileScanner.Split(bufio.ScanLines)
	var fileLines []string

	for fileScanner.Scan() {
		fileLines = append(fileLines, fileScanner.Text())
	}

	file.Close()

	fmt.Println(fileLines)

	// outputFile := "content/testWrite.txt"
	// f, err := os.Create(outputFile)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// defer f.Close()

	// _, error := f.WriteString(string(content))

	// if error != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Println("File has been created")

	// e := os.Rename(outputFile, "content/newFile.txt")
	// if e != nil {
	// 	log.Fatal(e)
	// }

	// fmt.Println("File has been renamed")
	
	email := string(content)
	fmt.Println(email + " validation: " + strconv.FormatBool(valid(email)))
	

	pathJSON := "content/JSONTEST.json"

	jsonContent, errJson := ioutil.ReadFile(pathJSON)
	if err != nil {
		log.Fatal("Error when opening JSON file: ", errJson)
	}

	var payload map[string]interface{}

	err = json.Unmarshal(jsonContent, &payload)

	if err != nil {
		log.Fatal("Error during Unmarshal():", err)
	}

	log.Printf("origin: %s\n", payload["origin"])
    log.Printf("user: %s\n", payload["user"])
    log.Printf("status: %t\n", payload["active"])

}
func valid(email string) bool {
	_, err := mail.ParseAddress(email)
	return err == nil
}

