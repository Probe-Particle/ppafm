#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C"
{
    int ReadNumsUpTo_C (char *fname, double *numbers, int * dims, int noline)
    {
        FILE *f;
        char line[5000]; // define a length which is long enough to store a line
        char *waste;
        int waste2;
        long i=0, j=0, k=0, tot=0; 

        int nx=dims[0];
        int ny=dims[1];
        int nz=dims[2];

        printf ("FileRead program: reading %s file\n", fname);
        printf ("XYZ dimensions are %d %d %d\n", dims[0], dims[1], dims[2]);
        f=fopen(fname, "r");
        if (f==NULL)
        {
            fprintf(stderr, "Can't open the file %s", fname);
            exit (1); 
        }
        for (i=0; i<noline; i++)
        {   
            waste=fgets(line,5000, f);
        }

//       printf ("Line: %s", line);
        for  (tot=0, k=0; k<dims[2]; k++)
        {
            for (j=0; j<dims[1]; j++)
            {   
                for (i=0; i<dims[0]; i++)
                {
                    waste2=fscanf(f,"%lf",&numbers[tot]);
//                    printf ("%20.20lf ", numbers[tot]);
                    tot++;
//                    if (tot > 5 ) exit(1);
                }
            }
        }

//       printf ("%lf %lf %lf %lf %lf\n", numbers[tot-1], numbers[tot-2], numbers[tot-3], numbers[tot-4], numbers[tot-5]);
        printf("Reading DONE\n");

        fclose(f);
        return 0;
    }

}
