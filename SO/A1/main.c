#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*A1. (2 puncte) Scrieti un program "copy" care se va lansa sub forma:
     copy   f1 + ... + fn    f
 (unde f1, ..., fn, f sunt fisiere) si are ca efect crearea lui f continand
 concatenarea lui f1, ..., fn; daca n=1 se copiaza f1 in f. Se vor folosi
 doar functiile de la nivelul inferior de prelucrare a fisierelor.*/

 // exemplu de rulare: ./main 1 1.txt 2.txt
 // copiaza 2.txt in 1.txt
int main(int argc, char** argv)
{
    //Introduceti numarul de fisiere care se vor copia
    int n = atoi(argv[1]);

    //specificatorul fisierului in care vrem sa copiem
    char pathb[256];
    strcpy(pathb, argv[2]);


    FILE *fa[n];

    for(int i = 0; i < n; i++){
        char patha[256];
        // specificatorul fisierului din care vrem sa copiem
        strcpy(patha,argv[i+3]);

        FILE *fa[i] = fopen(patha, "r+");
        FILE *fb = fopen(pathb, "a");

        if (fa[i] == NULL || fb == NULL ){
            puts("Fisierele nu pot fi deschise\n");
            exit(0);
        }
        char c;
        while ((c = fgetc(fa[i])) != EOF)
            fputc(c, fb);

        printf("Continutul fisierului a fost copiat.\n");

        fclose(fa[i]);
        fclose(fb);
    }

    return 0;
}
