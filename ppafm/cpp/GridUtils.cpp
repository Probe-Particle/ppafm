#include <stdio.h>
#include <stdlib.h>
//#include <string.h>

#include "Grid.h"

#include <locale.h>

#ifdef _WIN64 // Required for exports for ctypes on Windows
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT
#endif

GridShape gridShape;

// ==== teporary global for functions
double * data;

// for histogram ... would be better not make it global
namespace Histogram{
    Vec3d  center;
    double Htot;
    int    n;
    double dx;
    double * Hs;
    double * Ws;
}

extern "C" {

    DLLEXPORT int ReadNumsUpTo_C (char *fname, double *numbers, int * dims, int noline) {

        // Temporarily set fixed locale so that . (dot) is definitely the decimal separator
        setlocale(LC_NUMERIC, "C");

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
        if (f==NULL)        {
            fprintf(stderr, "Can't open the file %s", fname);
            exit (1);
        }
        for (i=0; i<noline; i++) {
            waste=fgets(line,5000, f);
        }
//       printf ("Line: %s", line);
        for  (tot=0, k=0; k<dims[2]; k++){
            for (j=0; j<dims[1]; j++){
                for (i=0; i<dims[0]; i++){
                    waste2=fscanf(f,"%lf",&numbers[tot]);
                    //printf ("%20.20lf ", numbers[tot]);
                    //printf ("%i %i %i %f \n", k, j, i, numbers[tot] );
                    tot++;
//                    if (tot > 5 ) exit(1);
                }
            }
        }
//       printf ("%lf %lf %lf %lf %lf\n", numbers[tot-1], numbers[tot-2], numbers[tot-3], numbers[tot-4], numbers[tot-5]);
        printf("Reading DONE\n");
        fclose(f);

        // Set locale back to the default locale
        setlocale(LC_NUMERIC, "");

        return 0;
    }

	DLLEXPORT void interpolate_gridCoord( int n, Vec3d * pos_list, double * data, double * out ){
		for( int i=0; i<n; i++ ){
			out[i] = interpolate3DWrap( data, gridShape.n, pos_list[i] );
		}
	}

	DLLEXPORT void interpolateLine_gridCoord( int n, Vec3d * p0, Vec3d * p1, double * data, double * out ){
        //printf( " interpolateLine n %i  p0 (%g,%g,%g) p1 (%g,%g,%g) \n", n,   p0->x,p0->y,p0->z, p1->x,p1->y,p1->z );
		Vec3d dp,p;
		dp.set_sub( *p1, *p0 );
		dp.mul( 1.0/n );
		p.set( *p0 );
        //printf( " interpolateLine n %i  p (%g,%g,%g) dp (%g,%g,%g) \n", n,   p.x,p.y,p.z, dp.x,dp.y,dp.z );
		for( int i=0; i<n; i++ ){
			//printf( " i, n  %i %i  pi0  %f %f %f  \n", i, n,  p.x,p.y,p.z );
			out[i] = interpolate3DWrap( data, gridShape.n, p );
			p.add( dp );
		}
	}

	DLLEXPORT void interpolateLine_cartes( int n, Vec3d * p0, Vec3d * p1, double * data, double * out ){
		Vec3d dp,p;
		dp.set_sub( *p1, *p0 );
		dp.mul( 1.0/n );
		p.set( *p0 );
		for( int i=0; i<n; i++ ){
			//printf( " i, n  %i %i  pi0  %f %f %f  \n", i, n,  p.x,p.y,p.z );
			Vec3d gp;
			gridShape.cartesian2grid( p, gp );
			//printf( "%i (%g,%g,%g) (%g,%g,%g) \n", i, p.x, p.y, p.z,  gp.x, gp.y, gp.z );
			out[i] = interpolate3DWrap( data, gridShape.n, gp );
			p.add( dp );
		}
	}

	DLLEXPORT void interpolateQuad_gridCoord( int * nij, Vec3d * p00, Vec3d * p01, Vec3d * p10, Vec3d * p11, double * data, double * out ){
		int ni = nij[0];
		int nj = nij[1];
        //printf( "gridShape.n %i %i %i \n", gridShape.n.x, gridShape.n.y, gridShape.n.z );
        //printf( "n (%i,%i) (%g,%g,%g) (%g,%g,%g) (%g,%g,%g) (%g,%g,%g) \n", nij[0],nij[1],   p00->x,p00->y,p00->z,   p01->x,p01->y,p01->z,   p10->x,p10->y,p10->z,   p11->x,p11->y,p11->z );
		Vec3d dpi0,dpi1,pi0,pi1;
		dpi0.set_sub( *p10, *p00 ); dpi0.mul( 1.0/ni );
		dpi1.set_sub( *p11, *p01 ); dpi1.mul( 1.0/ni );
		pi0.set( *p00 );
		pi1.set( *p01 );
        //printf( "(%g,%g,%g) (%g,%g,%g) \n",   dpi0.x,dpi0.y,dpi0.z, dpi1.x,dpi1.y,dpi1.z );
		for( int i=0; i<ni; i++ ){
			//printf( " i,ni,nj   %i %i %i  pi0  %f %f %f   pi1  %f %f %f \n", i, ni, nj,     pi0.x,pi0.y,pi0.z,     pi1.x,pi1.y,pi1.z );
			interpolateLine_gridCoord( nj, &pi0, &pi1, data, out + ( i * nj ) );
			pi0.add( dpi0 );
			pi1.add( dpi1 );
		}
	}

	// ---------  1D radial histogram
	inline void acum_sphere_hist( int ibuff, const Vec3d& pos, void * args ){
	    Vec3d dr;
	    dr.set_sub(pos, Histogram::center);
	    double r = dr.norm();
	    double u = r/Histogram::dx;
	    int i = (int)(u);
	    #if 0
        Histogram::Hs[i] += data[ibuff];
        Histogram::Ws[i] += 1;
        #else
        u = u-i;
        double h = data[ibuff];
        double mu = 1-u;
        Histogram::Hs[i  ] += mu*h;
        Histogram::Hs[i+1] += u*h;
        Histogram::Ws[i  ] += mu;
        Histogram::Ws[i+1] += u;
        #endif
    }
	DLLEXPORT void sphericalHist( double * data_, double* center, double dr, int n, double* Hs, double* Ws ){
	    data = data_; Histogram::n = n; Histogram::Hs=Hs; Histogram::Ws=Ws; Histogram::dx = dr; Histogram::center.set(center[0],center[1],center[2]);
        Vec3d r0; r0.set(0.0,0.0,0.0);
        interateGrid3D<acum_sphere_hist>( r0, gridShape.n, gridShape.dCell, NULL );
	}

	// ---------  find center of mass
	inline void acum_cog( int ibuff, const Vec3d& pos, void * args ){
	    double h = fabs( data[ibuff] );
	    Histogram::Htot +=  h;
	    Histogram::center.add_mul( pos, h );
	    //printf("acum_cog %i (%g,%g,%g) %g \n", ibuff, pos.x, pos.y, pos.z, h );
	    //if( ibuff > 100 ) exit(0);
    }
	DLLEXPORT double cog( double * data_, double* center ){
	    data = data_; Histogram::Htot += 0;  Histogram::center.set(0.0);
        Vec3d r0; r0.set(0.0,0.0,0.0);
        interateGrid3D<acum_cog>( r0, gridShape.n, gridShape.dCell, NULL );
        Histogram::center.mul( 1/Histogram::Htot  );
        ((Vec3d*)center)->set(Histogram::center);
        return Histogram::Htot;
	}

	DLLEXPORT void interpolate_cartesian( int n, Vec3d * pos_list, double * data, double * out ){
		for( int i=0; i<n; i++ ){
			Vec3d gpos;
			gridShape.cartesian2grid( pos_list[i], gpos );
			out[i] = interpolate3DWrap( data, gridShape.n, gpos );
		}
	}

	DLLEXPORT void setGridN( int * n ){
		//gridShape.n.set( *(Vec3i*)n );
		gridShape.n.set( n[2], n[1], n[0] );
		printf( " nxyz  %i %i %i \n", gridShape.n.x, gridShape.n.y, gridShape.n.z );
	}

	DLLEXPORT void setGridCell( double * cell ){
		gridShape.setCell( *(Mat3d*)cell );
        gridShape.printCell();
	}

}
