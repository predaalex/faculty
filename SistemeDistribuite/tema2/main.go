package main

import (
	"log"
	"strings"

	"github.com/chrislusf/glow/flow"
)

func main() {

	log.Print("EXERCITIUL 2")

	input := [][]string{
		{"a1551a", "parc", "ana", "minim", "1pcl3"},
		{"calabalac", "tivit", "leu", "zece10", "ploaie","9ana9"},
		{"lalalal", "tema", "papa", "ger"}}

	flow.New().Source(
		func(channel chan []string) {
			for _, line := range input {
				channel <- line
			}
		}, 3,
	).Map(func(line []string, ch chan string) {
		for _, word := range line {
			ch <- word
		}
	}).Filter(func(word string) bool {
		var reverse string;
		byte_str := []rune(word)
		for i, j := 0, len(byte_str)-1; i < j; i, j = i+1, j-1 {
			byte_str[i], byte_str[j] = byte_str[j], byte_str[i]
		}
		reverse = string(byte_str)
		return reverse == word;
	}).Map(func(word string) int {
		return 1
	}).Reduce(func(x int, y int) int {
		return x + y
	}).Map(func(x int) {
		log.Printf("%.3f", float64(x)/float64(len(input)))
	}).Run()

	log.Print("EXERCITIUL 4")

    vocale := "aeiouAEIOU"

    input = [][]string{
		{"ana", "parc", "impare", "era", "copil"},
		{"cer", "program", "leu", "alee", "golang","info"},
		{"inima", "impar", "apa", "eleve"}}

    flow.New().Source(
        func(channel chan []string) {
            for _, line := range input {
                channel <- line
            }
        }, 3,
    ).Map(func(line []string, ch chan string) {
        for _, word := range line {
            ch <- word
        }
    }).Filter(func(word string) bool {
		if(strings.Contains(vocale, word[:1]) && strings.Contains(vocale, word[len(word) - 1:])) {
			return true
		} else {
			return false
		}
    }).Map(func(word string) int {
        return 1
    }).Reduce(func(x int, y int) int {
        return x + y
    }).Map(func(x int) {
        log.Printf("%.3f", float64(x)/float64(len(input)))
    }).Run()
}

