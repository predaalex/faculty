package main

import (
	"bufio"
	"fmt"
	"net"
	"strconv"
	"strings"
	// "strings"
)

type Client struct {
	name string
	conn net.Conn
	request string
}

func reversedNumbers(data string) []int {
	reveredData := []int{}
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
	return reveredData
}

func clientResponse(clients chan Client) {

	for {
		client := <-clients
		
		name := client.name[:len(client.name) - 2]
		request := strings.Split(client.request[:len(client.request)-2], "|")
		cerinta := request[0]
		data := request[1]
		fmt.Println("Clientul cu numele " + name + 
		" a cerut rezolvarea " + cerinta + 
		" cu setul de date " + data +
		" s-a conectat.")

		switch cerinta {
		case "3":
			fmt.Println("Server is processing 3rd problem.")

			response := reversedNumbers(data)

			responseString := strings.Trim(strings.Replace(fmt.Sprint(response), " ", ",", -1), "[]")

			index := strings.LastIndex(responseString, ",")
			responseString = responseString[:index] + " cu suma " + responseString[index + 1:]
			
			fmt.Println("Server is sending to client the answer: " + responseString + ".")
			client.conn.Write([]byte(responseString + "."))
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