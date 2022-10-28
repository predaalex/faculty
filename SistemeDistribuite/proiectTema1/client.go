package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"time"
)

func main() {
	f, err := os.Open("fisier.txt")
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()

	reader := bufio.NewReader(f)

	name, err1 := reader.ReadString('\n')
	request, err2 := reader.ReadString('\n')

	if err1 != nil || err2 != nil {
		log.Fatalf("read file line error: %v | %v", err1, err2)
	}

	serverConnection, _ := net.Dial("tcp", "127.0.0.1:45600")

	fmt.Print("Numele cu care te conectezi la server este: " + name)
	fmt.Fprintf(serverConnection, name)

	
	fmt.Print("Requestul trimis este: " + request)
	fmt.Fprintf(serverConnection, request)

	cititorServer, _ := bufio.NewReader(serverConnection).ReadString('.')

	for start := time.Now(); time.Now().Sub(start) < time.Second; {
	}

	fmt.Println(cititorServer)


}