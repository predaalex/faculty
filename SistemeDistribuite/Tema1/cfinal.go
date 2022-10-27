package main

import (
	"bufio"
	"fmt"

	// "io/ioutil"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {

	/*
		//citirea din fisier
		continut, err := ioutil.ReadFile("fisier.txt")

		if err != nil {
			fmt.Println(err)
		}

		//conversia datelor din bytes
		container := string(continut)
		fmt.Println("container:", container )
		val1 := strings.Split(container, "\n")[0]
		//val1 = strings.Trim(val1, " ")
		lungimeMaxVector, _ := strconv.Atoi(val1)
		val2 := strings.Split(container, "\n")[1]

		//afisam
		fmt.Print("LungimeMaxVector:", val1)
		fmt.Print("AM citit din fisier val 2:", val2)
		fmt.Print("lungmaxvect:", lungimeMaxVector)
	*/

	//citire din fisier
	f, err := os.Open("fisier.txt")
	if err != nil {
		fmt.Println(err)
	}

	defer f.Close()

	scanner := bufio.NewScanner(f)

	var elems = []string{}
	var numereCitite = []int{}
	for scanner.Scan() {
		elems = append(elems, scanner.Text())
		numar, _ := strconv.Atoi(scanner.Text())
		numereCitite = append(numereCitite, numar)
	}

	for i := 0; i < len(numereCitite); i++ {
		fmt.Println(numereCitite[i] + 2)
	}
	var lungimeMaxVector = numereCitite[0]
	var nefolosit = numereCitite[1]

	fmt.Println("Am citit din fisier numarul maxim de elemente din array: ", lungimeMaxVector)
	fmt.Println("Am citit din fisier nr2: ", nefolosit)

	//Pasul 1 - ne conectam la server
	conexiune_server, _ := net.Dial("tcp", "127.0.0.1:45600")

	cititor := bufio.NewReader(os.Stdin)
	fmt.Print("Numele cu care vrei sa te conectezi la server: ")
	mesaj, _ := cititor.ReadString('\n')

	//Pasul 3 - trimiterea mesajului catre server
	fmt.Fprintf(conexiune_server, mesaj)
	fmt.Println("Ce cerinta individuala doresti si ce date vrei sa trimiti la server? ")
	fmt.Print("Ele trebuie despartite prin '|'. Introdu-le aici >> ")

	cerereClient, _ := cititor.ReadString('\n')
	//fmt.Println(cerereClient)
	//verificare date

	elementeDate := strings.Split(strings.Split(cerereClient, "|")[1], " ")
	if len(elementeDate) > lungimeMaxVector {
		fmt.Printf("Nu se pot trimite mai mult de %d elemente.\n", lungimeMaxVector)
		os.Exit(3)
	}
	fmt.Fprintf(conexiune_server, cerereClient)

	cititorServer, _ := bufio.NewReader(conexiune_server).ReadString('\n')

	for start := time.Now(); time.Now().Sub(start) < time.Second; {
	}

	fmt.Println(cititorServer)
	//mesajFinal := strings.Split(cititorServer, "=")
	// fmt.Fprintf(conexiune_server, "Clientul a primit raspuns "+strings.Split(mesajFinal[2], ":")[1]+"\n")
	// for i := 0; i < len(mesajFinal); i++ {
	// 	fmt.Println(mesajFinal[i])
	// }

}
