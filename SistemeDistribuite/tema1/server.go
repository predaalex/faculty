// programul se ruleaza cu go run ./server
package main

import (
	"bufio"
	"fmt"
	"net"
	"strconv"
	"strings"
)

type Client struct {
	name string
	conn net.Conn
	request string
}

func reversedNumbers(data string) string {
	var reveredData []int
	sum := 0

	dataArray := strings.Split(data, ",")

	for i := 0; i < len(dataArray); i++{
		number, err := strconv.Atoi(dataArray[i])
		if err != nil {
			fmt.Println("Error at data processing.")
		}
		
		reversedNumber := 0
		for number != 0 {
			reversedNumber = (reversedNumber * 10) + (number % 10)
			number /= 10
		}

		reveredData = append(reveredData, reversedNumber)
		sum += reversedNumber
	}

	reveredData = append(reveredData, sum)

	responseString := strings.Trim(strings.Replace(fmt.Sprint(reveredData), " ", ",", -1), "[]")

	index := strings.LastIndex(responseString, ",")
	responseString = responseString[:index] + " cu suma " + responseString[index + 1:]

	return responseString
}
func isPrime(number int) bool {
	if (number <= 1){
		return false
	} else {
		var i int
		for i = 2; i < number / 2; i++ {
			if (number % i == 0){
				return false
			}
		}
	}
	return true;
}
func figureNumber(number int) int {
	var counter int = 0
	for number != 0 {
		counter ++
		number /= 10
	}
	return counter
}

func figureCounter(data string) string {

	var numbersArray []int
	dataArray := strings.Split(data, ",")
	var figureCounter int = 0

	for i := 0; i < len(dataArray); i++ {
		number, err := strconv.Atoi(dataArray[i])
		if err != nil {
			fmt.Println("Error at data processing.")
		}
		
		if(isPrime(number)) {
			numbersArray = append(numbersArray, number)
			figureCounter += figureNumber(number)
		}
			

	}

	var responseData string
	var stringNumberArray string = strings.Trim(strings.Replace(fmt.Sprint(numbersArray), " ", ",", -1), "[]")

	responseData = strconv.Itoa(figureCounter) + " cifre(nr " + stringNumberArray + ")"

	return responseData
}

func decodedText(data string) string {

	var responseData string
	var i int = 0
	var apparitions int = 0

	for i = 0; i < len(data); i++ {

		if data[i] >= 48 && data[i] <= 57 {
			number, _ := strconv.Atoi(string(data[i]))
			apparitions = apparitions * 10 + number

		} else {
			j := 0
			for j = 0; j < apparitions; j++ {
				responseData += string(data[i])
			}
			apparitions = 0
		}

	}

	return responseData
}

func clientResponse(clients chan Client) {

	for {
		client := <-clients
		
		name := client.name[:len(client.name) - 2]
		request := strings.Split(client.request[:len(client.request)-2], "|")
		cerinta := request[0]
		data := request[1]
		fmt.Println("Client with name " + name + 
		" requested to solve problem " + cerinta + 
		" with set of data " + data +
		" has connected.")

		switch cerinta {
		case "3":
			fmt.Println("Server is processing 3rd problem.")

			response := reversedNumbers(data)

			fmt.Println("Server is sending to client the answer: " + response + ".")
			client.conn.Write([]byte(response + "."))
			break
		case "7":
			fmt.Println("Server is processing 7th problem.")

			response := decodedText(data)

			fmt.Println("Server is sending to client the answer: " + response + ".")
			client.conn.Write([]byte(response + "."))
			break
		case "8":
			fmt.Println("Server is processing 8th problem.")
			
			response := figureCounter(data)

			fmt.Println("Server is sending to client the answer: " + response + ".")
			client.conn.Write([]byte(response + "."))
			break
		default:
			client.conn.Write([]byte("The requested task is not available."))
			break
		}
	}
}

func main() {
	clients := make(chan Client)

	fmt.Println("Server started...")

	listen, _ := net.Listen("tcp", ":45600")

	for {
		
		connection, _ := listen.Accept()
		
		go clientResponse(clients) 

		go func() {
			for {
				name, err := bufio.NewReader(connection).ReadString('\n')
				request, err := bufio.NewReader(connection).ReadString('\n')

				if err != nil {
					fmt.Println("Client has disconnected.")
					break
				}

				clients <- Client{name, connection, request}
			}
		}()
	}
}