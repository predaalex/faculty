#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    // Verificăm dacă argumentul a fost furnizat în linia de comandă
    if (argc < 2) {
        printf("Introduceți un număr întreg ca argument.\n");
        return 1;
    }

    // Convertim argumentul din șir de caractere în număr întreg
//    int numar = atoi(argv[1]);

    char *ptr;
    long numar = strtol(argv[1], &ptr, 10);

//    argv[1] sir vid
//    ptr != null
    if ( *ptr != '\0' || strcmp(argv[1], "") == 0) {
        fprintf(stderr, "Invalid %s\n",argv[1]);
        exit(EXIT_FAILURE);
    }

    // Descompunem numărul în factori primi
    printf("%d = ", numar);
    for (int factor = 2; factor <= numar; factor++) {
        int putere = 0;
        while (numar % factor == 0) {
            putere++;
            numar /= factor;
        }
        if (putere > 0) {
            printf("%d", factor);
            if (putere > 1) {
                printf("^%d", putere);
            }
            if (numar > 1) {
                printf(" x ");
            }
        }
    }

    printf("\n");
    return 0;
}
