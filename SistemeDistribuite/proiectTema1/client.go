package main

import (
    "bufio"
    "fmt"
    "net/http"
)

func main() {
	resp, err := http.Get("http://localhost:8090/headers")

	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	fmt.Println("Response status:", resp.Status)

	scanner := bufio.NewScanner(resp.Body)
	// scanner.Scan()
	// fmt.Println(scanner.Text())
    for i := 0; scanner.Scan() && i < 2; i++ {
        fmt.Println(scanner.Text())
    }

    if err := scanner.Err(); err != nil {
        panic(err)
    }
}