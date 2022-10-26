package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"time"
)

// structura jsonului
type User struct {
	Id int
	Name string
	Password string
	LoggedAt time.Time
}

type Cafea struct {
	Id int
	Tip string
	Origine string
}

func writeJSON(cafele[] Cafea, path string){
	content, err := json.Marshal(cafele)
	if err != nil {
		fmt.Println(err)
	}

	err = ioutil.WriteFile(path, content, 0644)
	if err != nil {
		log.Fatal(err)
	}
}

func readJSON(path string) []Cafea{
	content, err := ioutil.ReadFile(path)

	if err != nil {
		log.Fatal(err)
	}

	var cafele []Cafea
	err = json.Unmarshal(content, &cafele)
	if err != nil {
		log.Fatal(err)
	}

	return cafele
}

func main() {
	fmt.Println("------------ Start of Execitiul 1 ------------")
	// populam cu informatii
	user0 := User{}
	user0.Id = 1
	user0.Name = "Alex"
	user0.Password = "Preda"
	user0.LoggedAt = time.Now()
	// encoding
	user0Byte, _ := json.Marshal(user0)

	// alt mod de a instanta un obiect
	user1 := User{2, "Achime", "Ionut", time.Now()}
	user1Byte, _ := json.Marshal(user1)

	fmt.Println("Encoding:")
	fmt.Println(string(user0Byte))
	fmt.Println(string(user1Byte))

	var user0Copie User
	json.Unmarshal(user0Byte, &user0Copie)

	var user1Copie User
	json.Unmarshal(user1Byte, &user1Copie)

	fmt.Println("Decoding:")
	fmt.Println(user0Copie)
	fmt.Println(user1Copie)
	fmt.Println("------------ End of Exercitiul 1 ------------")
	//////////////////////////////////////////////////////////////
	fmt.Println("------------ Start of Execitiul 2 ------------")
	path := "gigel.JSON"
	cafea1 := Cafea{1, "Jacobs", "Turcia"}
	cafea2 := Cafea{2, "Arabic", "Arabia"}

	var cafele []Cafea

	cafele = append(cafele, cafea1)
	cafele = append(cafele, cafea2)

	writeJSON(cafele, path)

	cafele = readJSON(path)

	fmt.Println("Cafele citite din JSON:")
	fmt.Println(cafele)

	fmt.Println("------------ End of Execitiul 2 ------------")
	/////////////////////////////////////////////////////////////
	fmt.Println("------------ Start of Execitiul 3 ------------")

	

	fmt.Println("------------ End of Execitiul 3 ------------")

}
