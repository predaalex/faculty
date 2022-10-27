package main

import (
	"bufio"
	"fmt"
	"math"
	"net"
	"regexp"
	"strconv"
	"strings"
	// "time"
	
)

/*
Each connection would have its own goroutine,
even though only one goroutine will be actively running at a time.
*/

type Client struct {
	name   string
	conn   net.Conn
	cerere string
}

func eBinar(numar int) bool {
	if numar <= 1 {
		return false
	}
	for numar != 0 {
		cifra := numar % 10
		if cifra != 1 && cifra != 0 {
			return false
		}
		numar = numar / 10
	}
	return true
}

//transformarea unui numar din binar in zecimal
func binarInDecimal(num int) int {
	var remainder int
	index := 0
	decimalNum := 0
	for num != 0 {
		remainder = num % 10
		num = num / 10
		decimalNum = decimalNum + remainder*int(math.Pow(2, float64(index)))
		index++
	}
	return decimalNum
}

func cerinta5(cuvinte []string) []int {
	rezultat := []int{}
	for i := 0; i < len(cuvinte); i++ {
		if regexp.MustCompile(`[a-zA-Z]`).MatchString(cuvinte[i]) {
			continue
		}
		if cuvinte[i] == "00" {
			rezultat = append(rezultat, 0)
			continue
		}
		if cuvinte[i] == "01" {
			rezultat = append(rezultat, 1)
			continue
		}
		numar, _ := strconv.Atoi(cuvinte[i])
		if eBinar(numar) {
			rezultat = append(rezultat, binarInDecimal(numar))
		}
	}
	return rezultat
}

func ePatratPerfect(x int) bool {
	if x >= 0 {

		extras := math.Sqrt(float64(x))
		if (extras * extras) == float64(x) {
			return true
		}
	}
	return false
}

func obtineNumar(cuv string) int {
	runeCuvant := []rune(cuv)
	var nr int = 0
	for i := 0; i < len(runeCuvant); i++ {
		c := string(runeCuvant[i])
		if val, err := strconv.Atoi(c); err == nil {
			nr = nr*10 + val
		}
	}
	return nr
}

func cerinta2(cuvinte []string) int {
	rez := 0
	for i := 0; i < len(cuvinte); i++ {
		num := obtineNumar(cuvinte[i])
		if ePatratPerfect(num) {
			rez = rez + 1
		}
	}
	return rez
}

func raspundeClient(clienti chan Client) {

	for {
		client := <-clienti
		
		// Send back the response.
		cerere := client.cerere[:len(client.cerere)-2]
		idCerinta := strings.TrimSpace(strings.Split(cerere, "|")[0]) //idCerinta := strings.Split(cerere, "|")[0]
		date := strings.Split(cerere, "|")[1]
		//fmt.Print("Clientul " +client.name + " a facut request cu datele " + date + " si cu cerinta individuala nr."+ idCerinta)
		numeEdit := client.name[:len(client.name)-2]
		fmt.Println("Clientul " + numeEdit + " s-a conectat")
		fmt.Println("A facut request cu datele:", date)
		fmt.Println(" si cu cerinta nr.", idCerinta)

		cuvinte := []string{}
		cuvinteDelimitate := strings.Split(date, " ")
		for i := 0; i < len(cuvinteDelimitate); i++ {
			if cuvinteDelimitate[i] != "" {
				cuvinte = append(cuvinte, cuvinteDelimitate[i])
			}
		}

		switch idCerinta {
		case "5":
			fmt.Println("A INTRAT la cerinta nr.5")
			rezultat := []int{}
			rezultat = cerinta5(cuvinte)
			//fmt.Println("cuvinte: ", len(cuvinte))
			//fmt.Println(":", rezultat)
			rezultatString := ""
			for i := 0; i < len(rezultat); i++ {
				rezultatString = rezultatString + " " + strconv.Itoa(rezultat[i])
			}
			//client.conn.Write([]byte(rezultatString))
			mesaj := "Serverul a calculat: raspunsul este " + rezultatString + "\n"
			client.conn.Write([]byte(mesaj))
			fmt.Println(mesaj)
			break
		case "2":
			fmt.Println("A INTRAT la cerinta nr.2")
			rezultat := cerinta2(cuvinte)
			rezultatString := strconv.Itoa(rezultat)
			//fmt.Println(":", rezultatString)
			mesaj := "Serverul a calculat: raspunsul este " + rezultatString + " numere\n"
			client.conn.Write([]byte(mesaj))
			fmt.Println(mesaj)
			break
		default:
			client.conn.Write([]byte("Nu exista implementare mom entan pentru cerinta aleasa."))
			break
		}
	}
}

func main() {

	clienti := make(chan Client)

	fmt.Println("Serverul a pornit...")

	//Pasul 1 - ascultam incoming-ul (conexiunile)
	listen, _ := net.Listen("tcp", ":45600")

	//Pasul 3 - rulam o bucla la infinit pentru conexiunile incoming (prin apasarea combinatiei de taste Ctrl+C)

	for {
		//Pasul 2 - acceptam conexiunile
		conexiune, _ := listen.Accept()
		go raspundeClient(clienti)

		go func() {

			for {
				nume, err := bufio.NewReader(conexiune).ReadString('\n')
				cerere, err := bufio.NewReader(conexiune).ReadString('\n')
				if err != nil {
					fmt.Printf("Clientul s-a deconectat.\n")
					break
				}

				clienti <- Client{nume, conexiune, cerere}
			}
		}()

	}
}
