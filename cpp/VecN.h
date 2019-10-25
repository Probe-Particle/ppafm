
#ifndef  VecN_h
#define  VecN_h

//#include <math.h>
//#include <cstdlib>
//#include <stdio.h>

//#include "fastmath.h"
#include  "fastmath_light.h"

namespace VecN{

    // =======  iterration over array

	inline double dot    (int n,const double* a,const double* b ){ double sum =0; for (int i=0; i<n; i++ ){ sum+= a[i]*b[i];             } return sum; }
	inline double sum    (int n,const double* a            ){ double sum =0; for (int i=0; i<n; i++ ){ sum+= a[i];                  } return sum; };
    inline double sum2   (int n,const double* a            ){ double sum =0; for (int i=0; i<n; i++ ){ double ai=a[i]; sum+=ai*ai;  } return sum; };
    inline double min    (int n,const double* a            ){ double amax=-1e+300; for (int i=0; i<n; i++ ){ double ai=a[i]; amax=fmax(amax,ai); } return amax; };
    inline double max    (int n,const double* a            ){ double amin=+1e+300; for (int i=0; i<n; i++ ){ double ai=a[i]; amin=fmin(amin,ai); } return amin; };
    inline double absmax (int n,const double* a            ){ double amax=0;       for (int i=0; i<n; i++ ){ double ai=a[i]; amax=fmax(amax,fabs(ai)); } return amax; };

    inline double minmax(int n,const double* a, double& vmin, double& vmax ){ vmin=+1e+300; vmax=-1e+300; for (int i=0; i<n; i++ ){ double ai=a[i]; vmin=_min(vmin,ai); vmax=_max(vmax,ai); } };

    inline int    imin(int n,const double* a ){ double amin=+1e+300; int im=-1; for (int i=0; i<n; i++ ){ double ai=a[i]; if(ai<amin){amin=ai;im=i;} } return im;  }
    inline int    imax(int n,const double* a ){ double amax=-1e+300; int im=-1; for (int i=0; i<n; i++ ){ double ai=a[i]; if(ai>amax){amax=ai;im=i;} } return im;  }

    inline double err2     (int n,const double* y1s,const double* y2s ){ double sum=0;        for (int i=0; i<n; i++ ){ double d=(y1s[i]-y2s[i]); sum+=d*d;          } return sum;  }
    inline double errAbsMax(int n,const double* y1s,const double* y2s ){ double amax=-1e+300; for (int i=0; i<n; i++ ){ double d=(y1s[i]-y2s[i]); amax=_max(d,amax); } return amax; }

    // =======  basic vector arritmetics

	inline void set( int n,const double  f,            double* out ){  	for (int i=0; i<n; i++ ){ out[i] = f;	      } }
	inline void add( int n,const double  f,const double* b, double* out ){  	for (int i=0; i<n; i++ ){ out[i] = f+b[i];    } }
	inline void mul( int n,const double  f,const double* b, double* out ){  	for (int i=0; i<n; i++ ){ out[i] = f*b[i];    } }

	inline void set( int n,const double* a,            double* out ){  	for (int i=0; i<n; i++ ){ out[i] = a[i];      } }
	inline void add( int n,const double* a,const double* b, double* out ){  	for (int i=0; i<n; i++ ){ out[i] = a[i]+b[i]; } }
	inline void sub( int n,const double* a,const double* b, double* out ){  	for (int i=0; i<n; i++ ){ out[i] = a[i]-b[i]; } }
	inline void mul( int n,const double* a,const double* b, double* out ){  	for (int i=0; i<n; i++ ){ out[i] = a[i]*b[i]; } }
	inline void div( int n,const double* a,const double* b, double* out ){  	for (int i=0; i<n; i++ ){ out[i] = a[i]/b[i]; } }
	inline void fma( int n,const double* a,const double* b, double f, double* out ){ for(int i=0; i<n; i++) { out[i]=a[i]+f*b[i]; }  }

	// =======  function-array operations

    inline double dot      (int n,const double* xs,const double* ys, Func1d func ){ double sum=0;        for (int i=0; i<n; i++ ){ sum+= ys[i]*func(xs[i]); }                         return sum;  }
    inline double err2     (int n,const double* xs,const double* ys, Func1d func ){ double sum=0;        for (int i=0; i<n; i++ ){ double d=(func(xs[i])-ys[i]); sum+=d*d;          } return sum;  }
    inline double errAbsMax(int n,const double* xs,const double* ys, Func1d func ){ double amax=-1e+300; for (int i=0; i<n; i++ ){ double d=(func(xs[i])-ys[i]); amax=_max(d,amax); } return amax; }
    inline double min (int n, double* a, Func1d func          ){ double amax=-1e+300; for (int i=0; i<n; i++ ){ double ai=a[i]; amax=_max(amax,ai); } return amax; };

    inline double set( int n,const double* xs,             Func1d func, double* out ){ for (int i=0; i<n; i++ ){ out[i] = func(xs[i]);       } };
    //inline void add  ( int n, double* xs, double* ys, Func1d func, double* out ){ for (int i=0; i<n; i++ ){ out[i] = ys[i]+func(xs[i]); } }
	//inline void sub  ( int n, double* xs, double* ys, Func1d func, double* out ){ for (int i=0; i<n; i++ ){ out[i] = ys[i]-func(xs[i]); } }
	//inline void mul  ( int n, double* xs, double* ys, Func1d func, double* out ){ for (int i=0; i<n; i++ ){ out[i] = ys[i]*func(xs[i]); } }
	//inline void div  ( int n, double* xs, double* ys, Func1d func, double* out ){ for (int i=0; i<n; i++ ){ out[i] = ys[i]/func(xs[i]); } }

    // =======  function-function operations

    inline double dot( int n,const double* xs, Func1d funcA, Func1d funcB ){ double sum=0; for (int i=0; i<n; i++ ){ double xi=xs[i]; sum+=funcA(xi)*funcB(xi); } return sum; }

	// =======  initialization and I/O

	inline void arange ( int n, double xmin, double dx  , double* out ){ double x=xmin; for(int i=0; i<n; i++){out[i]=x; x+=dx; } }
	inline void linspan( int n, double xmin, double xmax, double* out ){ double dx=(xmax-xmin)/n; arange( n, xmin, dx,out );    }

	inline void random_vector ( int n, double xmin, double xmax, double * out ){
		double xrange = xmax - xmin;
		for (int i=0; i<n; i++ ){		out[i] = xmin + xrange*randf();	}
	}

//	inline void print_vector( int n, double * a ){
//		for (int i=0; i<n; i++ ){	printf( "%f ", a[i] );	}
//		printf( "\n" );
//	}

}

#endif
