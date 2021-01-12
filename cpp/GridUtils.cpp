#include <stdio.h>
#include <stdlib.h>
//#include <string.h>

#include "Grid.h"

#include <locale.h> 

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

    int ReadNumsUpTo_C (char *fname, double *numbers, int * dims, int noline) {
    
        setlocale( LC_ALL, "C" ); // https://msdn.microsoft.com/en-us/library/x99tb11d(v=vs.71).aspx
        //setlocale( LC_ALL, "" );
        //setlocale( LC_ALL, "En_US" );  // to sove problem with ',' vs '.' caused by PyQt
        
        FILE *f;
        char line[5000]; // define a length which is long enough to store a line
        char *waste;
        int waste2;
        //int i=0, j=0, k=0, tot=0; 
        //int nx=dims[0];
        //int ny=dims[1];
        //int nz=dims[2];
        //printf ("FileRead program: reading %s file\n", fname);
        //printf ("XYZ dimensions are %i %i %i  noline %i \n", dims[0], dims[1], dims[2], noline );
        f=fopen(fname, "r");
        if (f==NULL)        {
            fprintf(stderr, "Can't open the file %s", fname);
            exit (1); 
        }
        for (int i=0; i<noline; i++) {   
            waste=fgets(line,5000, f);
			//printf ("waste[%i]: %s",i, line );	
        }
        //printf ("Line: %s", line);
		//int i=0;
		double nums[8];
		int ntot = dims[0] * dims[1] * dims[2]; 
		int itot=0;
		for(int il=0; il<ntot; il++){
			//waste=fgets(line,5000, f);
			//printf( "[%i,%i] >>%s<<\n", il, itot, line );
			int ngot = fscanf(f,"%lf %lf %lf %lf %lf %lf %lf %lf", &nums[0],&nums[1],&nums[2],&nums[3],&nums[4],&nums[5],&nums[6],&nums[7] );
			//int ngot = sscanf(line,"%lf %lf %lf %lf %lf %lf %lf %lf",              &nums[0],&nums[1],&nums[2],&nums[3],&nums[4],&nums[5],&nums[6],&nums[7] );
			//printf       ( "ngot %i \n", ngot );
			//printf       ( "[%i,%i] %g %g %g %g %g %g %g %g \n", il, itot, nums[0],nums[1],nums[2],nums[3],nums[4],nums[5],nums[6],nums[7] );
			for(int j=0; j<ngot; j++ ){
				numbers[itot]=nums[j];
				itot++;
			}
			if(itot>=ntot) break;
			//waste=fgets(line,5000, f);
			//waste2=fscanf(f,"%lf",&numbers[tot]);
			//printf ("line[%i] %s", i, line );
			//printf ("%20.20lf ", numbers[tot]);
			//printf ("%i %i %i %f \n", k, j, i, numbers[tot] );
			//if (il > 50 ) exit(1);
        }
//       printf ("%lf %lf %lf %lf %lf\n", numbers[tot-1], numbers[tot-2], numbers[tot-3], numbers[tot-4], numbers[tot-5]);
        //printf("Reading DONE\n");
        fclose(f);
        return 0;
    }

	void interpolate_gridCoord( int n, Vec3d * pos_list, double * data, double * out ){
		for( int i=0; i<n; i++ ){ 
			out[i] = interpolate3DWrap( data, gridShape.n, pos_list[i] ); 
		}
	}

	void interpolateLine_gridCoord( int n, Vec3d * p0, Vec3d * p1, double * data, double * out ){
        //printf( " interpolateLine n %i  p0 (%g,%g,%g) p1 (%g,%g,%g) \n", n,   p0->x,p0->y,p0->z, p1->x,p1->y,p1->z );
		Vec3d dp,p; 
		dp.set_sub( *p1, *p0 );
		dp.mul( 1.0d/n );
		p.set( *p0 );
        //printf( " interpolateLine n %i  p (%g,%g,%g) dp (%g,%g,%g) \n", n,   p.x,p.y,p.z, dp.x,dp.y,dp.z );
		for( int i=0; i<n; i++ ){ 
			//printf( " i, n  %i %i  pi0  %f %f %f  \n", i, n,  p.x,p.y,p.z );
			out[i] = interpolate3DWrap( data, gridShape.n, p );
			p.add( dp );
		}
	}

	void interpolateLine_cartes( int n, Vec3d * p0, Vec3d * p1, double * data, double * out ){
		Vec3d dp,p; 
		dp.set_sub( *p1, *p0 );
		dp.mul( 1.0d/n );
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

	void interpolateQuad_gridCoord( int * nij, Vec3d * p00, Vec3d * p01, Vec3d * p10, Vec3d * p11, double * data, double * out ){
		int ni = nij[0];
		int nj = nij[1];
        //printf( "gridShape.n %i %i %i \n", gridShape.n.x, gridShape.n.y, gridShape.n.z );
        //printf( "n (%i,%i) (%g,%g,%g) (%g,%g,%g) (%g,%g,%g) (%g,%g,%g) \n", nij[0],nij[1],   p00->x,p00->y,p00->z,   p01->x,p01->y,p01->z,   p10->x,p10->y,p10->z,   p11->x,p11->y,p11->z );
		Vec3d dpi0,dpi1,pi0,pi1;
		dpi0.set_sub( *p10, *p00 ); dpi0.mul( 1.0d/ni );
		dpi1.set_sub( *p11, *p01 ); dpi1.mul( 1.0d/ni );
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

	void stampToGrid2D( int* ns1_, int* ns2_, Vec2d* p0_, Vec2d* a_, Vec2d* b_, double* stamp, double* canvas, double coef ){
		Vec2i ns1 = *(Vec2i*)ns1_;
		Vec2i ns2 = *(Vec2i*)ns2_;
		Vec2d p0  = *(Vec2d*)p0_;
		Vec2d a   = *(Vec2d*)a_;
		Vec2d b   = *(Vec2d*)b_;
		//printf( "stampToGrid2D \n");
		for(int iy=0; iy<ns1.y; iy++){
			for(int ix=0; ix<ns1.x; ix++){
				double x = p0.x + a.x*ix + b.x*iy;
				double y = p0.y + a.y*ix + b.y*iy;
				int jx  = (int)x; 
				int jy  = (int)y; 
				double dx=x-jx; double mx=1.-dx;
				double dy=y-jy; double my=1.-dy;
				jx=wrap(jx,ns2.x);
				jy=wrap(jy,ns2.y);
				int jx1 = wrap(jx+1,ns2.x);
				int jy1 = wrap(jy+1,ns2.y);
				double v = stamp[iy*ns1.x+ix] * coef;
				//if( (jx<0)||(jx>=ns2.x)||(jy<0)||(jy>=ns2.y) ){
				//	printf( " %i %i -> %i %i %i %i  %g | %i %i \n", ix,iy, jx,jy,jx1,jy1, v,  ns2.x, ns2.y );
				//}
				canvas[jy *ns2.x+jx ] += v*mx*my;
				canvas[jy *ns2.x+jx1] += v*dx*my;
				canvas[jy1*ns2.x+jx ] += v*mx*dy;
				canvas[jy1*ns2.x+jx1] += v*dx*dy; 
			}
		}
		//printf( "stampToGrid2D DONE \n");
	}

    void stampToGrid2D_complex( int* ns1_, int* ns2_, Vec2d* p0_, Vec2d* a_, Vec2d* b_, double* stamp_, double* canvas_, Vec2d* coef_ ){
		Vec2i ns1  = *(Vec2i*)ns1_;
		Vec2i ns2  = *(Vec2i*)ns2_;
		Vec2d p0   = *(Vec2d*)p0_;
		Vec2d a    = *(Vec2d*)a_;
		Vec2d b    = *(Vec2d*)b_;
        Vec2d coef = *(Vec2d*)coef_;
        //printf( " sizeof(double) %i sizeof(Vec2d) %i \n", sizeof(double), sizeof(Vec2d) );
		//printf( "stampToGrid2D_complex START\n");
        Vec2d* stamp  = (Vec2d*)stamp_;
        Vec2d* canvas = (Vec2d*)canvas_;
        
		for(int iy=0; iy<ns1.y; iy++){
			for(int ix=0; ix<ns1.x; ix++){
				double x = p0.x + a.x*ix + b.x*iy;
				double y = p0.y + a.y*ix + b.y*iy;
				int jx  = (int)x; 
				int jy  = (int)y; 
				double dx=x-jx; double mx=1.-dx;
				double dy=y-jy; double my=1.-dy;
				jx=wrap(jx,ns2.x);
				jy=wrap(jy,ns2.y);
				int jx1 = wrap(jx+1,ns2.x);
				int jy1 = wrap(jy+1,ns2.y);
				Vec2d v = stamp[iy*ns1.x+ix];
                v.mul_cmplx(coef);
				//if( (jx<0)||(jx>=ns2.x)||(jy<0)||(jy>=ns2.y) ){
				//	printf( " %i %i -> %i %i %i %i  %g | %i %i \n", ix,iy, jx,jy,jx1,jy1, v,  ns2.x, ns2.y );
				//}
                //canvas[jy *ns2.x+jx ].x += v.x;
                //canvas[jy *ns2.x+jx ].y += v.y;
				canvas[jy *ns2.x+jx ].add_mul( v, mx*my );
				canvas[jy *ns2.x+jx1].add_mul( v, dx*my );
				canvas[jy1*ns2.x+jx ].add_mul( v, mx*dy );
				canvas[jy1*ns2.x+jx1].add_mul( v, dx*dy ); 
			}
		}
        
        /*
        int j;
        j=100; for(int i=0;i<ns2.x/2;i++){ canvas[i*ns2.x+j].x=1.0; canvas[i*ns2.x+j].y=2.0; };
        j=100; for(int i=0;i<ns2.x;i++){ canvas[j*ns2.x+i].x=1.0; canvas[j*ns2.x+i].y=2.0; };
		*/
        //printf( "stampToGrid2D_complex DONE \n");
	}

    /*
    inline void addTrilinar( const Vec3d& p, Vec3i ns, double val ){
        int jx  = (int)p.x; 
        int jy  = (int)p.y; 
        int jy  = (int)p.y;
        double dx=x-jx; double mx=1.-dx;
        double dy=y-jy; double my=1.-dy;
        double dz=z-jz; double mz=1.-dz;
        jx=wrap(jx,ns.x);
        jy=wrap(jy,ns.y);
        jz=wrap(jz,ns.y);
        int jx1 = wrap(jx+1,ns.x);
        int jy1 = wrap(jy+1,ns.y);
        int jy1 = wrap(jy+1,ns.y);
        //if( (jx<0)||(jx>=ns2.x)||(jy<0)||(jy>=ns2.y) ){
        //	printf( " %i %i -> %i %i %i %i  %g | %i %i \n", ix,iy, jx,jy,jx1,jy1, v,  ns2.x, ns2.y );
        //}
        //canvas[jy *ns2.x+jx ].x += v.x;
        //canvas[jy *ns2.x+jx ].y += v.y;
        canvas[jy *ns2.x+jx ].add_mul( v, mx*my );
        canvas[jy *ns2.x+jx1].add_mul( v, dx*my );
        canvas[jy1*ns2.x+jx ].add_mul( v, mx*dy );
        canvas[jy1*ns2.x+jx1].add_mul( v, dx*dy );
        canvas[jy1*ns2.x+jx ].add_mul( v, mx*dy );
        canvas[jy1*ns2.x+jx1].add_mul( v, dx*dy );
    }
    */

    inline void addTrilinar_v2d( const Vec3d& p, Vec3i ns, Vec2d val, Vec2d* canvas ){
        int jx  = (int)p.x; int jx1=jx+1; 
        int jy  = (int)p.y; int jy1=jy+1;
        int jz  = (int)p.z; int jz1=jz+1; 
        /*
        jx=wrap(jx,ns.x); int jx1 = wrap(jx+1,ns.x);
        jy=wrap(jy,ns.y); int jy1 = wrap(jy+1,ns.y);
        jz=wrap(jz,ns.z); int jz1 = wrap(jz+1,ns.z);
        if( (jx<0)||(jx>=ns.x) || (jy<0)||(jy>=ns.y) || (jz<0)||(jz>=ns.z) ){
        	printf( " (%i,%i,%i) (%i,%i,%i) \n", jx,jy,jz, ns.x, ns.y, ns.y );
            exit(0);
        }
        int nxy=ns.x*ns.y;
        int jzn =jz *nxy;
        int jzn1=jz1*nxy;
        int jyn =jy *ns.x;
        int jyn1=jy1*ns.x;
        */
        /*
        //if( (jx<10000) &&  (jy<10000) && (jz<2)  ){
        if( (jx>=0)&&(jx1<ns.x) &&  (jy>=0)&&(jy1<ns.y) && (jz>=0)&&(jz1<ns.z)  ){
            //int i=jz*nxy + jy*ns.x + jx;
            //int ntot = ns.x*ns.y*ns.z;
            //printf("%i %i %i | %i<%i \n", jx, jy, jz, i, ntot );
            //canvas[ jzn +jyn +jx ].add_mul( val, mx*my*mz );
            //canvas[ i ].add_mul( val, mx*my*mz );
            canvas[ jz*nxy + jy*ns.x + jx ].add_mul( val, mx*my*mz );
        }
        */
        if( (jx>=0)&&(jx1<ns.x) &&  (jy>=0)&&(jy1<ns.y) && (jz>=0)&&(jz1<ns.z)  ){
            int nxy=ns.x*ns.y;
            double dx=p.x-jx; double mx=1.-dx; 
            double dy=p.y-jy; double my=1.-dy;
            double dz=p.z-jz; double mz=1.-dz;
            int jzn =jz *nxy;
            int jzn1=jz1*nxy;
            int jyn =jy *ns.x;
            int jyn1=jy1*ns.x;
            canvas[ jzn +jyn +jx ].add_mul( val, mx*my*mz );
            canvas[jzn +jyn +jx1].add_mul( val, dx*my*mz );
            canvas[jzn +jyn1+jx ].add_mul( val, mx*dy*mz );
            canvas[jzn +jyn1+jx1].add_mul( val, dx*dy*mz );
            canvas[jzn1+jyn +jx ].add_mul( val, mx*my*dz );
            canvas[jzn1+jyn +jx1].add_mul( val, dx*my*dz );
            canvas[jzn1+jyn1+jx ].add_mul( val, mx*dy*dz );
            canvas[jzn1+jyn1+jx1].add_mul( val, dx*dy*dz );
        }
    }

    void stampToGrid3D_complex( int* ns1_, int* ns2_, double* p0_, double* rot_, double* stamp_, double* canvas_, Vec2d* coef_ ){
		Vec3i ns1  = *(Vec3i*)ns1_;
		Vec3i ns2  = *(Vec3i*)ns2_;
		Vec3d p0   = *(Vec3d*)p0_;
		Mat3d rot  = *(Mat3d*)rot_;
        Vec2d coef = *(Vec2d*)coef_;
        //printf( " sizeof(double) %i sizeof(Vec2d) %i \n", sizeof(double), sizeof(Vec2d) );
		//printf( "stampToGrid2D_complex START\n");
        Vec2d* stamp  = (Vec2d*)stamp_;
        Vec2d* canvas = (Vec2d*)canvas_;
        //printf( "ns1 %i %i %i \n", ns1.x, ns1.y, ns1.z );
        //printf( "ns2 %i %i %i \n", ns2.x, ns2.y, ns2.z );
        //printf( "p0 %g %g %g \n", p0.x, p0.y, p0.z );
        //printf( "a %g %g %g \n", rot.a.x, rot.a.y, rot.a.z );
        //printf( "b %g %g %g \n", rot.b.x, rot.b.y, rot.b.z );
        //printf( "c %g %g %g \n", rot.c.x, rot.c.y, rot.c.z );
        int nxy1= ns1.x*ns1.y;
        int nxy2= ns2.x*ns2.y;
        /*
        for(int iz=0; iz<ns2.z; iz++){
            for(int iy=0; iy<ns2.y; iy++){
                for(int ix=0; ix<ns2.x; ix++){
                    //int i1 =  iz*(ns1.x*ns1.y) + iy*ns1.x + ix;
                    int i2 =  iz*(ns2.x*ns2.y) + iy*ns2.x + ix;
                    //canvas[i2].add( stamp[i1] );
                    //canvas[i2].add( 1.0 );
                    //printf("%i %i %i \n", jx, jy, jz );
                    addTrilinar_v2d( {ix,iy,iz}, ns2, {1.,1.}, canvas );
                }
            }
        }
        */
        for(int iz=0; iz<ns1.z; iz++){
            for(int iy=0; iy<ns1.y; iy++){
                for(int ix=0; ix<ns1.x; ix++){
                    Vec3d p; rot.dot_to_T( (Vec3d){ix,iy,iz}, p);
                    p.add(p0);
                    Vec2d val = stamp[ iz*nxy1 + iy*ns1.x + ix];
                    val.mul_cmplx(coef);
                    addTrilinar_v2d( p, ns2, val, canvas );
                    //addTrilinar_v2d( p, ns2, {1.,1.}, canvas );
                } 
            }
        }
        /*
        for(int iz=0; iz<ns1.z; iz++){
            for(int iy=0; iy<ns1.y; iy++){
                for(int ix=0; ix<ns1.x; ix++){
                    int i1 =  iz*(ns1.x*ns1.y) + iy*ns1.x + ix;
                    int i2 =  iz*(ns2.x*ns2.y) + iy*ns2.x + ix;
                    canvas[i2].add( stamp[i1] );
                    //canvas[i2].add( 1.0 );
                }
            }
        }
        */
        //int iy,iz,ix;
        //int ntot= ns2.x*ns2.y*ns2.z;
        //for(int i=0;i<ntot;i++){ canvas[i].x=i%ns2.z; canvas[i].y=0; };
        //for(int i=0;i<ntot;i++){ canvas[i].x=i%(ns2.y*ns2.z); canvas[i].y=0; };
        //for(int i=0;i<ntot;i++){ canvas[i].x=i%(ns2.y*ns2.z); canvas[i].y=0; };
        //iy=10; iz=10; for(int ix=0;ix<ns2.x;ix++){ int i= iz*nxy2 + iy*ns2.x + ix; canvas[i].x=1.0; canvas[i].y=2.0; };
        //ix=10; iz=10; for(int iy=0;iy<ns2.y;iy++){ int i= iz*nxy2 + iy*ns2.x + ix; canvas[i].x=1.0; canvas[i].y=2.0; };
        //ix=10; iy=10; for(int iz=0;iz<ns2.z;iz++){ int i= iz*nxy + iy*ns2.x + ix; canvas[i].x=1.0; canvas[i].y=2.0; };
        //j=100; for(int i=0;i<ns2.y;i++){ canvas[j*ns2.x+i].x=1.0; canvas[j*ns2.x+i].y=2.0; };
        //j=100; for(int i=0;i<ns2.z;i++){ canvas[j*ns2.x+i].x=1.0; canvas[j*ns2.x+i].y=2.0; };
        //printf( "stampToGrid2D_complex DONE \n");
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
	void sphericalHist( double * data_, double* center, double dr, int n, double* Hs, double* Ws ){ 
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
	double cog( double * data_, double* center ){ 
	    data = data_; Histogram::Htot += 0;  Histogram::center.set(0.0);
        Vec3d r0; r0.set(0.0,0.0,0.0);
        interateGrid3D<acum_cog>( r0, gridShape.n, gridShape.dCell, NULL );
        Histogram::center.mul( 1/Histogram::Htot  );
        ((Vec3d*)center)->set(Histogram::center);
        return Histogram::Htot;
	}

	void interpolate_cartesian( int n, Vec3d * pos_list, double * data, double * out ){
		for( int i=0; i<n; i++ ){ 
			Vec3d gpos;
			gridShape.cartesian2grid( pos_list[i], gpos );
			out[i] = interpolate3DWrap( data, gridShape.n, gpos ); 
		}
	}

	void setGridN( int * n ){
		//gridShape.n.set( *(Vec3i*)n );
		gridShape.n.set( n[2], n[1], n[0] );
		printf( " nxyz  %i %i %i \n", gridShape.n.x, gridShape.n.y, gridShape.n.z );
	}

	void setGridCell( double * cell ){
		gridShape.setCell( *(Mat3d*)cell );
        gridShape.printCell();
	}

}


