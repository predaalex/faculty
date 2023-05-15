#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    int p=1;
    // printf("Introduceti numarul ");
    // scanf("%d",&a);
    char *aux = argv[1];
    int a = atoi(aux);
    for(int d=2;d<=a;d++)
    {p=0;
        if(a%d==0)
        {
            while(a%d==0)
            {
               a/=d;
               p++;
            }
            printf("%d",d);
            printf("^%d", p);
            printf("\n");
        }
    }
    return 0;
}
