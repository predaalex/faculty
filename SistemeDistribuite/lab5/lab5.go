package main

import (
	"io/ioutil"
	"encoding/json"
	"fmt"
	"log"
)
type Cantitati struct {
	Mere float32
	Pere float32
	Piersici float32
	Capsuni float32
}

func readJSON(path string) Cantitati {
	content, err := ioutil.ReadFile(path)

	if err != nil {
		log.Fatal(err)
	}

	var cantitati Cantitati

	err = json.Unmarshal(content, &cantitati)

	if err != nil {
		log.Fatal(err)
	}

	return cantitati
}

func main() {

	fmt.Println("------------ Start of Execitiul 1 ------------")

	path := "cantitati_fructe.JSON"

	var cant Cantitati

	cant = readJSON(path)

	fmt.Print("Mere:")
	fmt.Println(cant.Mere)

	fmt.Print("Pere:")
	fmt.Println(cant.Pere)

	fmt.Print("Piersici:")
	fmt.Println(cant.Piersici)

	fmt.Print("Capsuni:")
	fmt.Println(cant.Capsuni)

	fmt.Println("------------ End of Execitiul 1 ------------")
}