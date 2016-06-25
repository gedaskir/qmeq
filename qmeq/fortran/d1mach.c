#include <stdio.h>
#include <float.h>
#include <math.h>
double d1mach_(long *i)
{
 switch(*i){
   case 1: return DBL_MIN;
   case 2: return DBL_MAX;
   case 3: return DBL_EPSILON/FLT_RADIX;
   case 4: return DBL_EPSILON;
   case 5: return log10((double)FLT_RADIX);
   }
 fprintf(stderr, "invalid argument: d1mach(%ld)\n", *i);
 //exit(1); 
 return 0; /* some compilers demand return values */
}
