package main

import (
	"log"
	"strconv"
	"strings"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"
)

func operation(var1 string, op string) int {
	values := strings.Split(var1, "\n")

	int1, _ := strconv.Atoi(values[0])
	int2, _ := strconv.Atoi(values[1])


	switch op {
	case "+":
		return int1 + int2
	case "-":
		return int1 - int2
	case "X":
		return int1 * int2
	case "/":
		if int2 == 0 {
			log.Fatal("Divide by 0!!!")
			return 0
		}
		return int1 / int2
	default:
		return 0
	}

}

func main() {
	myApp := app.New()
	myWindow := myApp.NewWindow("Calculator")

	value := ""
	op := ""
	var result int
	keyboardNumbers := container.New(layout.NewGridWrapLayout(fyne.NewSize(50, 50)))


	text1 := widget.NewButton("1", func() {
		value += "1"
	})
	text2 := widget.NewButton("2", func() {
		value += "2"
	})
	text3 := widget.NewButton("3", func() {
		value += "3"
	})
	text4 := widget.NewButton("4", func() {
		value += "4"
	})
	text5 := widget.NewButton("5", func() {
		value += "5"
	})
	text6 := widget.NewButton("6", func() {
		value += "6"
	})
	text7 := widget.NewButton("7", func() {
		value += "7"
	})
	text8 := widget.NewButton("8", func() {
		value += "8"
	})
	text9 := widget.NewButton("9", func() {
		value += "9"
	})
	textx := widget.NewButton("X", func() {
		op += "X"
		value += "\n"
	})
	textminus := widget.NewButton("-", func() {
		op += "-"
		value += "\n"
	})
	textplus := widget.NewButton("+", func() {
		op += "+"
		value += "\n"
	})
	textslash := widget.NewButton("/", func() {
		op += "/"
		value += "\n"
	})
	text0 := widget.NewButton("0", func() {
		value += "0"
	})
	textc := widget.NewButton("C", func() {
		value = ""
		op = ""
	})
	textresult := widget.NewLabel(strconv.Itoa(result))

	textegal := widget.NewButton("=", func() {
		result := operation(value, op)
		log.Println(result)
		op = "="
		textresult.SetText(strconv.Itoa(result))   
	})
	

	keyboardNumbers = container.New(layout.NewGridWrapLayout(fyne.NewSize(50, 50)),
		text7, text8, text9, textslash, text4, text5, text6, textx,
		text1, text2, text3, textminus, textc, text0, textplus, textegal, textresult)
		
	myWindow.SetContent(keyboardNumbers)

	myWindow.Resize(fyne.NewSize(250, 100))
	myWindow.ShowAndRun()
}
