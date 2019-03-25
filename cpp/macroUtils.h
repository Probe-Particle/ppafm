
#ifndef  macroUtils_h
#define  macroUtils_h

#define SWAP( a, b, TYPE ) { TYPE t = a; a = b; b = t; }

#define _max(a,b)      ((a>b)?a:b)
#define _min(a,b)      ((a<b)?a:b)
#define _abs(a)        ((a>0)?a:-a)
//#define _clamp(x,a,b)  max(a, min(b, x))
//#define _clamp(n, lower, upper) if (n < lower) n= lower; else if (n > upper) n= upper
#define _clamp(a, lower, upper) ((a>lower)?((a<upper)?a:upper):lower)

#define _minit( i, x, imin, xmin )  if( x<xmin ){ xmin=x; imin=i; }
#define _maxit( i, x, imax, xmax )  if( x>xmax ){ xmax=x; imax=i; }

#define _setmin( xmin, x )  if( x<xmin ){ xmin=x; }
#define _setmax( xmax, x )  if( x>xmax ){ xmax=x; }

#define _circ_inc( i, n )   i++; if(i>=n) i=0;
#define _circ_dec( i, n )   i--; if(i< 0) i=n-1;

//#define _realloc(TYPE,arr,n){ if(var) delete [] arr; arr=new TYPE[n]; }

template<typename T> inline void _realloc(T*& arr, int n){ if(arr) delete [] arr; arr=new T[n]; }
template<typename T> inline void _dealloc(T*& arr       ){ if(arr) delete [] arr; arr=0; }

#endif
