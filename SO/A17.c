#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
    char buffer1[1024], buffer2[1024];
    char *new_pwd = "C:\\Users\\allex\\Downloads\\DDD_sem2_2022_2023"; // schimba cu directorul dorit

    // obtinem directorul curent
    if (getcwd(buffer1, sizeof(buffer1)) == NULL) {
        perror("getcwd() error");
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
    if (getcwd(buffer2, sizeof(buffer2)) == NULL) {
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
