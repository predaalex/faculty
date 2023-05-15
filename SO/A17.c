#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
    size_t size = PATH_MAX; // marimea maxima a path ului

    char *buffer1 = NULL, *buffer2 = NULL;

    buffer1 = malloc(size * sizeof(char));
    buffer2 = malloc(size * sizeof(char));

    if (buffer1 == NULL || buffer2 == NULL) {
        perror("Failed to alocate memory\n");
        exit(EXIT_FAILURE);
    }

    char *new_pwd = "C:\\Users\\allex\\Downloads\\353_Preda_Alexandru-Florin.zip\\353_Preda_Alexandru-Florin"; // schimba cu directorul dorit


    // obtinem directorul curent
    if (getcwd(buffer1, size) == NULL) {
        perror("getcwd() error");
        exit(EXIT_FAILURE);
    }

    if (strcmp(buffer1, new_pwd) == 0) {
        perror("path-ul nou este acelasi cu cel curent");
        exit(EXIT_FAILURE);
    }

    // afisam directorul curent
    printf("Director curent initial: %s\n", buffer1);

    // modificam variabila de mediu PWD
    char envvar[1024];
    snprintf(envvar, sizeof(envvar), "PWD=%s", new_pwd);
    if (putenv(envvar) != 0) {
        perror("putenv() error");
        exit(EXIT_FAILURE);
    }

    // obtinem noul director curent
    if (getcwd(buffer2, size) == NULL) {
        perror("getcwd() error");
        exit(EXIT_FAILURE);
    }

    // afisam noul director curent
    printf("Director curent dupa schimbarea PWD: %s\n", buffer2);

    // comparam cele doua buffere si afisam mesajul corespunzator
    if (strcmp(buffer1, buffer2) == 0) {
        printf("Directorul curent ramane neschimbat.\n");
    } else {
        printf("Directorul curent s-a schimbat.\n");
    }

    return 0;
}
