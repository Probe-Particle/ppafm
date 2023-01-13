
#define R2SAFE          1e-4f
#define COULOMB_CONST   14.399644f  // [eV*Ang/e^2]

//#define N_RELAX_STEP_MAX  64
#define N_RELAX_STEP_MAX  16
#define F2CONV  1e-8

#include "splines.cl"

inline float3 rotMat ( float3 v,  float3 a, float3 b, float3 c  ){ return (float3)(dot(v,a),dot(v,b),dot(v,c)); }
inline float3 rotMatT( float3 v,  float3 a, float3 b, float3 c  ){ return a*v.x + b*v.y + c*v.z; }
inline int mod(int x, int y) {
    int rem = (int) x % y;
    return (rem >= 0 ? rem : rem + y);
}

// Do accurate trilinear interpolation of a buffer array. Buffer should use C memory layout.
// The cartesian position is transformed to the grid coordinate indices using the coordinate transformation
// matrix whose rows are T_A, T_B, and T_C.
float linearInterpB(float3 pos, float3 origin, float3 T_A, float3 T_B, float3 T_C, int3 nGrid, __global float *buf) {

    // Find coordinate index (ijk) that is just before the position and figure out
    // how far past the voxel coordinate we are (d).
    float3 ijk0;
    float3 coord = pos - origin;
    float3 coordT = (float3)(dot(coord, T_A), dot(coord, T_B), dot(coord, T_C));
    float3 d = fract(coordT, &ijk0);

    // Find values at all the corners next to the main voxel (periodic boundary conditions)
    int nyz = nGrid.y * nGrid.z;
    int i0 = mod((int) ijk0.x, nGrid.x);
    int j0 = mod((int) ijk0.y, nGrid.y);
    int k0 = mod((int) ijk0.z, nGrid.z);
    int i1 = (i0 + 1) % nGrid.x;
    int j1 = (j0 + 1) % nGrid.y;
    int k1 = (k0 + 1) % nGrid.z;
    float c000 = buf[i0 * nyz + j0 * nGrid.z + k0];
    float c001 = buf[i0 * nyz + j0 * nGrid.z + k1];
    float c010 = buf[i0 * nyz + j1 * nGrid.z + k0];
    float c011 = buf[i0 * nyz + j1 * nGrid.z + k1];
    float c100 = buf[i1 * nyz + j0 * nGrid.z + k0];
    float c101 = buf[i1 * nyz + j0 * nGrid.z + k1];
    float c110 = buf[i1 * nyz + j1 * nGrid.z + k0];
    float c111 = buf[i1 * nyz + j1 * nGrid.z + k1];

    // Interpolate
    float c = (1 - d.x) * (1 - d.y) * (1 - d.z) * c000
            + (1 - d.x) * (1 - d.y) *      d.z  * c001
            + (1 - d.x) *      d.y  * (1 - d.z) * c010
            + (1 - d.x) *      d.y  *      d.z  * c011
            +      d.x  * (1 - d.y) * (1 - d.z) * c100
            +      d.x  * (1 - d.y) *      d.z  * c101
            +      d.x  *      d.y  * (1 - d.z) * c110
            +      d.x  *      d.y  *      d.z  * c111;

    return c;

}

// Same as linearInterpB, except for float4 type
float4 linearInterpB4(float3 pos, float3 origin, float3 T_A, float3 T_B, float3 T_C, int3 nGrid, __global float4 *buf) {

    // Find coordinate index (ijk) that is just before the position and figure out
    // how far past the voxel coordinate we are (d).
    float3 ijk0;
    float3 coord = pos - origin;
    float3 coordT = (float3)(dot(coord, T_A), dot(coord, T_B), dot(coord, T_C));
    float3 d = fract(coordT, &ijk0);

    // Find values at all the corners next to the main voxel (periodic boundary conditions)
    int nyz = nGrid.y * nGrid.z;
    int i0 = mod((int) ijk0.x, nGrid.x);
    int j0 = mod((int) ijk0.y, nGrid.y);
    int k0 = mod((int) ijk0.z, nGrid.z);
    int i1 = (i0 + 1) % nGrid.x;
    int j1 = (j0 + 1) % nGrid.y;
    int k1 = (k0 + 1) % nGrid.z;
    float4 c000 = buf[i0 * nyz + j0 * nGrid.z + k0];
    float4 c001 = buf[i0 * nyz + j0 * nGrid.z + k1];
    float4 c010 = buf[i0 * nyz + j1 * nGrid.z + k0];
    float4 c011 = buf[i0 * nyz + j1 * nGrid.z + k1];
    float4 c100 = buf[i1 * nyz + j0 * nGrid.z + k0];
    float4 c101 = buf[i1 * nyz + j0 * nGrid.z + k1];
    float4 c110 = buf[i1 * nyz + j1 * nGrid.z + k0];
    float4 c111 = buf[i1 * nyz + j1 * nGrid.z + k1];

    // Interpolate
    float4 c = (1 - d.x) * (1 - d.y) * (1 - d.z) * c000
             + (1 - d.x) * (1 - d.y) *      d.z  * c001
             + (1 - d.x) *      d.y  * (1 - d.z) * c010
             + (1 - d.x) *      d.y  *      d.z  * c011
             +      d.x  * (1 - d.y) * (1 - d.z) * c100
             +      d.x  * (1 - d.y) *      d.z  * c101
             +      d.x  *      d.y  * (1 - d.z) * c110
             +      d.x  *      d.y  *      d.z  * c111;

    return c;

}

float4 getCoulomb( float4 atom, float3 pos ){
     float3  dp  =  pos - atom.xyz;
     float   ir2 = 1.0f/( dot(dp,dp) +  R2SAFE );
     float   ir  = sqrt(ir2);
     float   E   = atom.w*sqrt(ir2);
     return (float4)(dp*(E*ir2), E );
}

float4 getLJ( float3 apos, float2 cLJ, float3 pos ){
     float3  dp  =  pos - apos;
     float   ir2 = 1.0f/( dot(dp,dp) + R2SAFE );
     float   ir6 = ir2*ir2*ir2;
     float   E   =  (    cLJ.y*ir6 -   cLJ.x )*ir6;
     float3  F   = (( 12.0f*cLJ.y*ir6 - 6.0f*cLJ.x )*ir6*ir2)*dp;
     return (float4)(F, E);
}

float4 getMorse( float3 dp, float3 REA ){
    //float3  dp  =  pos - apos;
    float   r     = sqrt( dot(dp,dp) + R2SAFE );
    float   expar = exp( REA.z*(r-REA.x) );
    float   E     = REA.y*expar*( expar - 2 );
    float   fr    = REA.y*expar*( expar - 1 )*2*REA.z;
    return (float4)(dp*(fr/r), E);
}

float8 getLJC( float4 atom, float2 cLJ, float3 pos ){
     float3  dp  =  pos - atom.xyz;
     float   ir2 = 1.0/( dot(dp,dp) +  R2SAFE );
     float   ir6 = ir2*ir2*ir2;
     float   ELJ =  (    cLJ.y*ir6 -   cLJ.x )*ir6;
     float3  FLJ = (( 12.0f*cLJ.y*ir6 - 6.0f*cLJ.x )*ir6*ir2)*dp;
     float   ir  = sqrt(ir2);
     float   Eel = atom.w*sqrt(ir2);
     return (float8)(FLJ, ELJ, dp*(Eel*ir2), Eel );
}

float getLorenz( float4 atom, float4 coefs, float3 pos ){
     float3  dp  =  pos - atom.xyz;
     return coefs.x/( dot(dp,dp) +  coefs.y*coefs.y );
     //return 1.0/( dot(dp,dp) +  0.000 );
}


__kernel void evalCoulomb(
    int nAtoms,
    __global float4* atoms,
    __global float4*  poss,
    __global float4*    FE
){
    __local float4 LATOMS[32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float4 fe  = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        // if(iG==8485) printf("%i (%f,%f,%f)  %f \n", i, atoms[i].x, atoms[i].y, atoms[i].z, atoms[i].w );
        if(i < nAtoms) {
            LATOMS[iL] = atoms[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if(i0+j < nAtoms) {
                fe += getCoulomb( LATOMS[j], pos );
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = fe*COULOMB_CONST;
    //FE[iG] = poss[iG];
}

__kernel void evalLJ(
    const int nAtoms,
    __global float4* atoms,
    __global float2*  cLJs,
    __global float4*  poss,
    __global float4*    FE
){
    __local float4 LATOMS[32];
    __local float2 LCLJS  [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float4 fe  = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        if(i>=nAtoms) break;
        //if(iL==0) printf("%i (%f,%f,%f)  %f \n", i, atoms[i].x, atoms[i].y, atoms[i].z, atoms[i].w );
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            fe += getLJ( LATOMS[j].xyz, LCLJS [j], pos );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = fe;
    //FE[iG] = poss[iG];
}

__kernel void evalLJ_noPos(
    const int nAtoms,
    __global float4* atoms,
    __global float2*  cLJs,
    __global float4*    FE,
    int4 nGrid,
    float4 grid_p0,
    float4 grid_dA,
    float4 grid_dB,
    float4 grid_dC
){
    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    const int nab = nGrid.x*nGrid.y;
    const int ia  = iG%nGrid.x;
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  = iG/nab;
    const int nMax = nab*nGrid.z;

    if(iG>nMax) return;

    float3 pos = grid_p0.xyz + grid_dA.xyz*ia + grid_dB.xyz*ib  + grid_dC.xyz*ic;
    float4 fe  = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float4 xyzq = LATOMS[j];
                fe += getLJ(xyzq.xyz, LCLJS[j], pos);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    FE[iG] = fe;

}

__kernel void evalLJC_Q(
    const int nAtoms,
    __global float4* atoms,
    __global float2*  cLJs,
    __global float4*  poss,
    __global float4*    FE,
    float Qmix
){
    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float8 fe  = (float8) (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    //if(iG==0){ printf("evalLJC: nAtoms: %i \n", nAtoms ); }
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ) fe += getLJC( LATOMS[j], LCLJS[j], pos );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // http://www.informit.com/articles/article.aspx?p=1732873&seqNum=3
    //fe.hi  = fe.hi*COULOMB_CONST;
    Qmix *= COULOMB_CONST;
    FE[iG] = fe.lo + Qmix * fe.hi;
    //FE[iG] = poss[iG];
}

__kernel void evalLJC_Q_noPos(
    const int nAtoms,
    __global float4* atoms,
    __global float2*  cLJs,
    __global float4*    FE,
    int4 nGrid,
    float4 grid_p0,
    float4 grid_dA,
    float4 grid_dB,
    float4 grid_dC,
    float Qmix
){
    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    //float3 pos = poss[iG].xyz;
    //float3 pos = grid_p0 + grid_dA*get_global_id(0) + grid_dA*get_global_id(1)  + grid_dA*get_global_id (2);      // there would be more problematic local_id optimization

    const int nab = nGrid.x*nGrid.y;
    const int ia  = iG%nGrid.x;
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  = iG/nab;
    const int nMax = nab*nGrid.z;

    if(iG>nMax) return;

    float3 pos    = grid_p0.xyz + grid_dA.xyz*ia + grid_dB.xyz*ib  + grid_dC.xyz*ic;

    // if(iG==430662){
    //     printf("evalLJC_Q_noPos: nAtom %i nGrid(%i,%i,%i,%i)   nL %i  nG %i nG_ %i  \n", nAtoms, nGrid.x, nGrid.y, nGrid.z, nGrid.w, nL, get_global_size(0), nGrid.x*nGrid.y*nGrid.z );
    //     printf("evalLJC_Q_noPos: grid.p0 (%g,%g,%g) \n", grid_p0.x, grid_p0.y, grid_p0.z );
    //     printf("evalLJC_Q_noPos: grid.p0 (%g,%g,%g) \n", grid_dA.x, grid_dA.y, grid_dA.z );
    //     printf("evalLJC_Q_noPos: grid.p0 (%g,%g,%g) \n", grid_dB.x, grid_dB.y, grid_dB.z );
    //     printf("evalLJC_Q_noPos: grid.p0 (%g,%g,%g) \n", grid_dC.x, grid_dC.y, grid_dC.z );
    //     // 150*150*19 + 150*21 + 12 = 430662
    //     printf("evalLJC_Q_noPos: iG %i ia,ib,ic(%i,%i,%i)   pos(%g,%g,%g) \n", iG, ia, ib, ic,   pos.x,pos.y,pos.z  );
    // }

    float8 fe  = (float8) (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ) fe += getLJC( LATOMS[j], LCLJS[j], pos );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // http://www.informit.com/articles/article.aspx?p=1732873&seqNum=3
    //fe.hi  = fe.hi*COULOMB_CONST;

    //if(iG==430662){      printf("evalLJC_Q_noPos: iG %i ia,ib,ic(%i,%i,%i)   pos(%g,%g,%g) Qmix %g fe.lo(%g,%g,%g)  fe.hi(%g,%g,%g) \n", iG, ia, ib, ic,   pos.x,pos.y,pos.z,   Qmix,   fe.lo.x,fe.lo.y,fe.lo.z,   fe.hi.x,fe.hi.y,fe.hi.z  ); }

    Qmix *= COULOMB_CONST;
    FE[iG] = fe.lo + Qmix * fe.hi;
    //FE[iG] = poss[iG];
}

__kernel void evalLJC_QZs_noPos(
    const int nAtoms,
    __global float4* atoms,
    __global float2*  cLJs,
    __global float4*    FE,
    int4 nGrid,
    float4 grid_p0,
    float4 grid_dA,
    float4 grid_dB,
    float4 grid_dC,
    float4 Qs,
    float4 QZs
){
    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    const int nab = nGrid.x*nGrid.y;
    const int ia  = iG%nGrid.x;
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  = iG/nab;
    const int nMax = nab*nGrid.z;

    if(iG>nMax) return;
    //if(iG==0) printf( " Qs (%g,%g,%g,%g) QZs (%g,%g,%g,%g) \n", Qs.x,Qs.y,Qs.z,Qs.w,   QZs.x,QZs.y,QZs.z,QZs.w   );

    float3 pos    = grid_p0.xyz + grid_dA.xyz*ia + grid_dB.xyz*ib  + grid_dC.xyz*ic;

    float4 fe  = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

    Qs *= COULOMB_CONST;

    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                //fe += getLJC( LATOMS[j], LCLJS[j], pos );
                float4 xyzq = LATOMS[j];
                fe += getLJ     ( xyzq.xyz, LCLJS[j], pos );
                fe += getCoulomb( xyzq, pos+(float3)(0,0,QZs.x) ) * Qs.x;
                fe += getCoulomb( xyzq, pos+(float3)(0,0,QZs.y) ) * Qs.y;
                fe += getCoulomb( xyzq, pos+(float3)(0,0,QZs.z) ) * Qs.z;
                fe += getCoulomb( xyzq, pos+(float3)(0,0,QZs.w) ) * Qs.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //if ( (ia==75)&&(ib==75) ) { printf(" iz %i fe %g,%g,%g,%g \n", ic, fe.x, fe.y, fe.z, fe.w ); }

    FE[iG] = fe;
}

__kernel void evalLJC(
    int nAtoms,
    __global float4*   atoms,
    __global float2*    cLJs,
    __global float4*    poss,
    __global float8*    FE
){
    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float8 fe  = (float8) (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    //if(iG==0){ printf("evalLJC: nAtoms: %i \n", nAtoms ); }
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ) fe += getLJC( LATOMS[j], LCLJS[j], pos );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // http://www.informit.com/articles/article.aspx?p=1732873&seqNum=3
    fe.hi  = fe.hi*COULOMB_CONST;
    FE[iG] = fe;
}

__kernel void evalLJC_QZs(
    int nAtoms,
    __global float4*   atoms,
    __global float2*    cLJs,
    __global float4*    poss,
    __global float8*    FE,
    float4 Qs,
    float4 QZs
){
    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float8 fe  = (float8) (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    //if(iG==0){ printf("evalLJC: nAtoms: %i \n", nAtoms ); }
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            //if( (j+i0)<nAtoms ) fe += getLJC( LATOMS[j], LCLJS[j], pos );
            if( (j+i0)<nAtoms ){
                float4 xyzq = LATOMS[j];
                fe.lo += getLJ     ( xyzq.xyz, LCLJS[j], pos );
                fe.hi += getCoulomb( xyzq, pos+(float3)(0,0,QZs.x) ) * Qs.x;
                fe.hi += getCoulomb( xyzq, pos+(float3)(0,0,QZs.y) ) * Qs.y;
                fe.hi += getCoulomb( xyzq, pos+(float3)(0,0,QZs.z) ) * Qs.z;
                fe.hi += getCoulomb( xyzq, pos+(float3)(0,0,QZs.w) ) * Qs.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // http://www.informit.com/articles/article.aspx?p=1732873&seqNum=3
    fe.hi  = fe.hi*COULOMB_CONST;
    FE[iG] = fe;
}

// Add Lennard-Jones force to an existing forcefield grid. The forcefield values are saved in
// Fortran order. Local work group size can be at most 64.
__kernel void addLJ(
    const int nAtoms,       // Number of atoms
    __global float4* atoms, // Atom positions
    __global float2*  cLJs, // Lennard-Jones parameters for atoms
    __global float4*    FE, // Forcefield grid
    int4 nGrid,             // Grid size
    float4 grid_origin,     // Real-space origin of grid
    float4 grid_stepA,      // Real-space step sizes of grid lattice vectors
    float4 grid_stepB,
    float4 grid_stepC
){

    __local float4 LATOMS[64];
    __local float2 LCLJS [64];
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    int ind = get_global_id(0);
    if (ind > nGrid.x * nGrid.y * nGrid.z) return;

    // Convert linear index to x,y,z indices (Fortran order)
    int i = ind % nGrid.x;
    int j = (ind / nGrid.x) % nGrid.y;
    int k = ind / (nGrid.y * nGrid.x);

    // Calculate position in grid
    float3 pos = grid_origin.xyz + grid_stepA.xyz*i + grid_stepB.xyz*j  + grid_stepC.xyz*k;

    // Compute Lennard-Jones force from all of the atoms
    float4 fe = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    for (int i0 = 0; i0 < nAtoms; i0 += nL) {
        int ia = i0 + iL;
        LATOMS[iL] = atoms[ia];
        LCLJS [iL] = cLJs[ia];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int ja = 0; (ja < nL) && (ja < (nAtoms - i0)); ja++){
            fe += getLJ(LATOMS[ja].xyz, LCLJS[ja], pos);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Add to forcefield grid
    FE[ind] += fe;
}

// Compute Lennard Jones force field at grid points and add to it the electrostatic force
// from an electric field precomputed from a Hartree potential. The output buffer is
// written in Fortran memory layout in order to be compatible with OpenCL image
// read functions in subsequent steps.
__kernel void evalLJC_Hartree(
    const int nAtoms,           // Number of atoms
    __global float4* atoms,     // Positions of atoms
    __global float2* cLJs,      // Lennard-Jones parameters
    __global float4* E_field,   // Electric field
    __global float4* FE,        // Output force and energy
    int4 nGrid,                 // Size of grid
    float4 grid_origin,         // Real-space origin of grid
    float4 grid_stepA,          // Real-space step sizes of grid lattice vectors
    float4 grid_stepB,
    float4 grid_stepC,
    float4 T_A,                 // Rows of the transformation matrix for grid lattice coordinates
    float4 T_B,
    float4 T_C,
    float4 Qs,                  // Tip charges
    float4 QZs                  // Tip charge positions on z axis relative to PP
){

    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    int ind = get_global_id(0);
    if (ind > nGrid.x * nGrid.y * nGrid.z) return;

    // Convert linear index to x,y,z indices (C order)
    int i = ind / (nGrid.y * nGrid.z);
    int j = (ind / nGrid.z) % nGrid.y;
    int k = ind % nGrid.z;

    // Calculate position in grid
    float3 pos = grid_origin.xyz + grid_stepA.xyz*i + grid_stepB.xyz*j  + grid_stepC.xyz*k;

    // Compute Lennard-Jones force from all of the atoms
    float4 fe = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    for (int i0 = 0; i0 < nAtoms; i0 += nL) {
        int ia = i0 + iL;
        LATOMS[iL] = atoms[ia];
        LCLJS [iL] = cLJs[ia];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int ja = 0; (ja < nL) && (ja < (nAtoms - i0)); ja++){
            fe += getLJ(LATOMS[ja].xyz, LCLJS[ja], pos);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Add electrostatic contribution
    fe += Qs.x * linearInterpB4(pos + (float3)(0, 0, QZs.x), grid_origin.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid.xyz, E_field);
    fe += Qs.y * linearInterpB4(pos + (float3)(0, 0, QZs.y), grid_origin.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid.xyz, E_field);
    fe += Qs.z * linearInterpB4(pos + (float3)(0, 0, QZs.z), grid_origin.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid.xyz, E_field);
    fe += Qs.w * linearInterpB4(pos + (float3)(0, 0, QZs.w), grid_origin.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid.xyz, E_field);

    // Save to output buffer (Fortran order)
    FE[i + j * nGrid.x + k * nGrid.x * nGrid.y] = fe;

}

// Obtain electric field as the negative gradient of Hartree potential via centered difference
__kernel void gradPotential(
    __global float  *pot,   // Input Hartree potential
    __global float4 *field, // Output electric field
    int4            nGrid,  // Size of Hartree potential grid (x, y, z, _)
    float4          step,   // Step size of grid (x, y, z, _)
    int             C_order // If non-zeo, values are saved in C order, otherwise Fortran order
) {

    int ind = get_global_id(0);
    if (ind >= nGrid.x * nGrid.y * nGrid.z) return;

    // Get x,y,z indices
    int nyz = nGrid.y * nGrid.z;
    int i = ind / nyz;
    int j = (ind % nyz) / nGrid.z;
    int k = ind % nGrid.z;

    // Find global indices of adjacent points on each side.
    // On edges use periodic boundary conditions.
    int ip = i > 0           ? ind - nyz     : ind + (nGrid.x - 1) * nyz;
    int jp = j > 0           ? ind - nGrid.z : ind + (nGrid.y - 1) * nGrid.z;
    int kp = k > 0           ? ind - 1       : ind + (nGrid.z - 1);
    int in = i < nGrid.x - 1 ? ind + nyz     : ind - (nGrid.x - 1) * nyz;
    int jn = j < nGrid.y - 1 ? ind + nGrid.z : ind - (nGrid.y - 1) * nGrid.z;
    int kn = k < nGrid.z - 1 ? ind + 1       : ind - (nGrid.z - 1);

    // Compute value of field as centered difference. Also copy potential to
    // last place to be consistent with the other Coulomb kernels.
    float4 f = (float4)(
        0.5*(pot[ip] - pot[in]) / step.x,
        0.5*(pot[jp] - pot[jn]) / step.y,
        0.5*(pot[kp] - pot[kn]) / step.z,
        pot[ind]
    );

    // Save to output array
    int out_ind = C_order ? ind : i + nGrid.x * j + nGrid.x * nGrid.y * k;
    field[out_ind] = f;

}

// Obtain electric field as the negative gradient of Hartree potential via centered difference
// on a target grid that may be different from the grid of the Hartree potential
__kernel void gradPotentialGrid(
    __global float  *pot,   // Input Hartree potential.
    __global float4 *field, // Output electric field.
    int4   nGrid_H,         // Size of Hartree potential grid
    float4 T_A,             // Rows of the transformation matrix for Hartree potential grid lattice coordinates
    float4 T_B,
    float4 T_C,
    float4 origin_H,        // Real-space origin of Hartree potential grid
    int4   nGrid_T,         // Size of target grid
    float4 step_T_A,        // Real-space step sizes of target grid lattice vectors
    float4 step_T_B,
    float4 step_T_C,
    float4 origin_T,        // Real-space origin of target grid
    float4 h                // Finite-difference step
) {

    int ind = get_global_id(0);
    if (ind >= nGrid_T.x * nGrid_T.y * nGrid_T.z) return;

    // Convert linear index to x,y,z indices
    int i = ind / (nGrid_T.y * nGrid_T.z);
    int j = (ind / nGrid_T.z) % nGrid_T.y;
    int k = ind % nGrid_T.z;

    // Calculate position in target grid
    float3 pos = origin_T.xyz + i * step_T_A.xyz + j * step_T_B.xyz + k * step_T_C.xyz;

    // Interpolate potential at points around the center point
    float pot_   = linearInterpB(pos,                              origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);
    float pot_ip = linearInterpB(pos + (float3)(-h.x,    0,    0), origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);
    float pot_jp = linearInterpB(pos + (float3)(   0, -h.y,    0), origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);
    float pot_kp = linearInterpB(pos + (float3)(   0,    0, -h.z), origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);
    float pot_in = linearInterpB(pos + (float3)( h.x,    0,    0), origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);
    float pot_jn = linearInterpB(pos + (float3)(   0,  h.y,    0), origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);
    float pot_kn = linearInterpB(pos + (float3)(   0,    0,  h.z), origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);

    // Compute value of field as centered difference. Also copy potential to
    // last place to be consistent with the other Coulomb kernels.
    field[ind].x = 0.5*(pot_ip - pot_in) / h.x;
    field[ind].y = 0.5*(pot_jp - pot_jn) / h.y;
    field[ind].z = 0.5*(pot_kp - pot_kn) / h.z;
    field[ind].w = pot_;

}

// Interpolate an array at points specified by a grid. The grid can be rotated around a point.
__kernel void interp_at(
    __global float *in,     // Input array
    __global float *out,    // Output array
    int4   nGrid_in,        // Size of input array
    float4 T_A,             // Rows of the transformation matrix for input array lattice coordinates
    float4 T_B,
    float4 T_C,
    float4 origin_in,       // Real-space origin of input array
    int4   nGrid_out,       // Size of target grid
    float4 step_out_A,      // Real-space step sizes of output array lattice vectors
    float4 step_out_B,
    float4 step_out_C,
    float4 origin_out,      // Real-space origin of output array
    float4 rot_A,           // Rows of rotation matrix to apply
    float4 rot_B,
    float4 rot_C,
    float4 rot_center       // Point around which rotation is performed
){

    int ind = get_global_id(0);
    if (ind >= nGrid_out.x * nGrid_out.y * nGrid_out.z) return;

    // Convert linear index to x,y,z indices of output array
    int i = ind / (nGrid_out.y * nGrid_out.z);
    int j = (ind / nGrid_out.z) % nGrid_out.y;
    int k = ind % nGrid_out.z;

    // Calculate position in target grid
    float4 pos = origin_out + i * step_out_A + j * step_out_B + k * step_out_C;
    float4 d_pos = pos - rot_center;
    pos = rot_center + (float4)(dot(rot_A, d_pos), dot(rot_B, d_pos), dot(rot_C, d_pos), 0.0f);

    // Interpolate
    out[ind] = linearInterpB(pos.xyz, origin_in.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_in.xyz, in);

}

float3 tipForce( float3 dpos, float4 stiffness, float4 dpos0 ){
    float r = sqrt( dot( dpos,dpos) );
    return  (dpos-dpos0.xyz) * stiffness.xyz        // harmonic 3D
         + dpos * ( stiffness.w * (r-dpos0.w)/r );  // radial
}


__kernel void relaxStrokesDirect(
    int nAtoms,
    __global float4*    atoms,
    __global float2*    cLJs,
    __global float4*    points,
    __global float4*    FEs,
    float4 dTip,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    float Q,
    int nz
){

    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);


    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos    = tipPos.xyz + dpos0.xyz;

    const float dt   = relax_params.x;
    const float damp = relax_params.y;
    //printf( " %li (%f,%f,%f)  \n",  get_global_id(0), tipPos.x, tipPos.y, tipPos.z);

    for(int iz=0; iz<nz; iz++){
        float4 fe_;
        float3 vel   = 0.0f;


        for(int i=0; i<N_RELAX_STEP_MAX; i++){
            //fe        = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );

            // Get Atomic Forces
            float8 fe  = (float8) (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

            for (int i0=0; i0<nAtoms; i0+= nL ){
                int i = i0 + iL;
                //if(i>=nAtoms) break;  // wrong !!!!
                LATOMS[iL] = atoms[i];
                LCLJS [iL] = cLJs[i];
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int j=0; j<nL; j++){
                    if( (j+i0)<nAtoms ) fe += getLJC( LATOMS[j], LCLJS[j], pos );
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            fe_ = fe.lo + fe.hi * Q;


            float3 f  = fe_.xyz;
            f        += tipForce( pos-tipPos, stiffness, dpos0 );
            vel      *=       damp;
            vel      += f   * dt;
            pos.xyz  += vel * dt;
            if(dot(f,f)<F2CONV) break;

        }

        /*
            // Get Atomic Forces
            float8 fe  = (float8) (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

            for (int i0=0; i0<nAtoms; i0+= nL ){
                int i = i0 + iL;
                //if(i>=nAtoms) break;  // wrong !!!!
                LATOMS[iL] = atoms[i];
                LCLJS [iL] = cLJs[i];
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int j=0; j<nL; j++){
                    if( (j+i0)<nAtoms ) fe += getLJC( LATOMS[j], LCLJS[j], pos );
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        fe_ = fe.lo + fe.hi * -0.1f;
        */

        FEs[get_global_id(0)*nz + iz] = fe_;
        //FEs[get_global_id(0)*nz + iz].xyz = pos;
        tipPos += dTip.xyz;
        pos    += dTip.xyz;

    }



}




__kernel void evalMorse(
    const int nAtoms,
    __global float4*   atoms,
    __global float4*   REAs,
    __global float4*   poss,
    __global float4*   FE
){
    __local float4 lATOMs[32];
    __local float4 lREAs [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float4 fe  = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break; // wrong !!!!
        lATOMs[iL] = atoms[i];
        lREAs [iL] = REAs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ) fe += getMorse( pos - lATOMs[j].xyz, lREAs[j].xyz );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // http://www.informit.com/articles/article.aspx?p=1732873&seqNum=3
    FE[iG] = fe;
}


__kernel void evalLorenz(
    int nAtoms,
    __global float4*    atoms,
    __global float4*    coefs,
    __global float4*    poss,
    __global float*     FE
){
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float fe = 0.0f;

    //if(iG==0){
    //    printf( "nAtoms %i \n", nAtoms );
    //}
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){

            //if(iG==64*128 + 64) printf( "iatom %i atom (%g,%g,%g) pos(%g,%g,%g) \n", i0+j, LATOMS[j].x, LATOMS[j].y, LATOMS[j].z, pos.x,pos.y,pos.z );
            //if( (j+i0)<nAtoms ) fe += getLorenz( LATOMS[j], coefs[j], pos );
            if( (j+i0)<nAtoms ) fe += getLorenz( LATOMS[j], LCOEFS[j], pos );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = fe;
    //FE[iG] = pos.z;
}



__kernel void evalDisk(
    int nAtoms,
    __global float4*    atoms,
    __global float4*    coefs,
    __global float4*    poss,
    __global float*     FE,
    float dzmax,
    float dzmax_s,
    float offset,
    float Rpp,
    float4 rotA,
    float4 rotB,
    float4 rotC
){
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float fe = 0.0f;
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf( " xyzq (%g,%g,%g,%g) coef (%g,%g,%g,%g) \n", atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w,   coefs[i].x,coefs[i].y,coefs[i].z,coefs[i].w );  } }
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp  =  pos - LATOMS[j].xyz;
                float3 abc = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                //float   R  = coefs[j].z;
                float   R  = LCOEFS[j].z;
                float   r2 = dot(abc.xy,abc.xy);
                float dxy2 = r2/( (R*R) );
                float  z   = abc.z - offset;
                float  z_s = abc.z - LCOEFS[j].w - Rpp;
                if( ( dxy2 < 1.0f ) && ( z < dzmax ) && ( z_s < dzmax_s ) ){
                    //fe += 1-dxy2;
                    //fe += ( 1-(abc.z/dzmax) ) * sqrt( 1- dxy2 );
                    fe += ( 1.0f-(z/dzmax) ) * ( 1.0f- sqrt(dxy2) );
                    //fe += coefs[j].w;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = fe;
}

__kernel void evalSpheres(
    int nAtoms,
    __global float4*    atoms,
    __global float4*    coefs,
    __global float4*    poss,
    __global float*     FE,
    float Rpp,
    float zmin,
    float4 rotA,
    float4 rotB,
    float4 rotC
){
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;

    //float Rpp  =  1.0;
    //float Rpp  =  0.0;
    //float Rpp  = -0.7;

    float mask = 1.0;
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf( " xyzq (%g,%g,%g,%g) coef (%g,%g,%g,%g) \n", atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w,   coefs[i].x,coefs[i].y,coefs[i].z,coefs[i].w );  } }

    float ztop = zmin;
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp    = pos - LATOMS[j].xyz;
                float3 abc   = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                //float  Rvdw  = coefs[j].w + Rpp;
                float  Rvdw  = LCOEFS[j].w + Rpp;
                //float  Rvdw  = coefs[j].w;
                //float  Rvdw  = coefs[j].z;
                float r2xy   =  dot(abc.xy,abc.xy);
                float  z     = -abc.z + sqrt( Rvdw*Rvdw - r2xy );
                if(z>ztop){
                    ztop=z;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = ztop;
}


__kernel void evalSphereCaps(
    int nAtoms,
    __global float4*    atoms,
    __global float4*    coefs,
    __global float4*    poss,
    __global float*     FE,
    float Rpp,
    float zmin,
    float tgMax,
    float tgWidth,
    float4 rotA,
    float4 rotB,
    float4 rotC
){
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;

    //float Rpp  =  1.0;
    //float Rpp  =  0.0;
    //float Rpp  = -0.7;

    //float mask = 1.0;
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf( " xyzq (%g,%g,%g,%g) coef (%g,%g,%g,%g) \n", atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w,   coefs[i].x,coefs[i].y,coefs[i].z,coefs[i].w );  } }

    float ztop  = zmin;
    //float ztop  = 0;
    float tgtop = -1.0;
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp    = pos - LATOMS[j].xyz;
                float3 abc   = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                //float  Rvdw  = coefs[j].w + Rpp;
                float  Rvdw  = LCOEFS[j].w + Rpp;
                float r2xy   =  dot(abc.xy,abc.xy);

                float dz     =  sqrt( Rvdw*Rvdw - r2xy );
                float  z     = -abc.z + dz;

                //float tgWidth = 0.1f;
                float tg      = sqrt(r2xy)/Rvdw;
                z  = zmin + (z-zmin)*(1.0f-smoothstep( tgMax-tgWidth, tgMax, tg ));
                ztop = fmax( z, ztop );
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = ztop;

}


__kernel void evalDisk_occlusion(
    int nAtoms,
    __global float4*    atoms,
    __global float4*    coefs,
    __global float4*    poss,
    __global float*     FE,
    float Rpp,
    float zmin,
    float zmargin,
    float dzmax,
    float4 rotA,
    float4 rotB,
    float4 rotC
){
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;

    //float dzmax = 0.2;
    //float Rpp  =  1.0;
    //float Rpp  =  0.0;
    //float Rpp  = -0.7;

    float mask = 1.0;
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf( " xyzq (%g,%g,%g,%g) coef (%g,%g,%g,%g) \n", atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w,   coefs[i].x,coefs[i].y,coefs[i].z,coefs[i].w );  } }

    float ztop = zmin;
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp    = pos - LATOMS[j].xyz;
                float3 abc   = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                //float  Rvdw  = coefs[j].w + Rpp;
                float  Rvdw  = LCOEFS[j].w + Rpp;
                float  r2xy   =  dot(abc.xy,abc.xy);
                float z = 2.0*Rvdw - r2xy/(Rvdw*Rvdw) - abc.z; // osculation parabola to atom
                if(z>ztop){
                    ztop=z;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //FE[iG] = ztop;

    float fe   = 0.0f;
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp    = pos - LATOMS[j].xyz;
                float3 abc   = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                //float  Rvdw  = coefs[j].w + Rpp;
                float  Rvdw  = LCOEFS[j].w + Rpp;
                float  r2xy   =  dot(abc.xy,abc.xy);
                float z   = 2.0f*Rvdw - r2xy/(Rvdw*Rvdw) - abc.z; // osculation parabola to atom
                float cdz = 1.0 - ((ztop-z)/zmargin);
                //fe += cdz;
                //fe = fmax(ztop,z);
                //fe = fmax(cdz,fe);
                if( cdz>0.0f ){
                    //float   R     = coefs[j].z * 2.0;
                    float   R     = LCOEFS[j].z * 2.0;
                    float dxy2 = r2xy/( R*R );
                    if( ( dxy2 < 1.0f ) && ( abc.z < dzmax ) ){
                        //fe += 1-dxy2;
                        //fe += ( 1-(abc.z/dzmax) ) * sqrt( 1- dxy2 );
                        fe += ( 1.0f-(abc.z/dzmax) ) * ( 1.0f- sqrt(dxy2) ) * cdz;
                        //fe += coefs[j].w;
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = fe;


}



__kernel void evalMultiMapSpheres(
    int nAtoms,
    __global float4*    atoms,
    __global float4*    coefs,
    __global float4*    poss,
    __global float*     MultMap,
    float Rpp,
    float zmin,
    float4 rotA,
    float4 rotB,
    float4 rotC,
    int bOccl,
    int   nChan,
    float Rmin,
    float Rstep
){
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    float ztops[20];

    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;

    float mask = 1.0;
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf( " xyzq (%g,%g,%g,%g) coef (%g,%g,%g,%g) \n", atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w,   coefs[i].x,coefs[i].y,coefs[i].z,coefs[i].w );  } }

    float ztop = zmin;
    for (int i=0; i<nChan; i++){
        ztops[i]=zmin;
    }

    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp    = pos - LATOMS[j].xyz;
                float3 abc   = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                float  Rvdw  = LCOEFS[j].w + Rpp;
                float r2xy   =  dot(abc.xy,abc.xy);
                float  z     = -abc.z + sqrt( Rvdw*Rvdw - r2xy );

                int ityp = (int)( (LCOEFS[j].w - Rmin)/Rstep );
                if(ityp>=nChan)ityp=nChan-1;
                if(z>ztop){
                    ztop=z;
                }
                if(z>ztops[ityp]){
                    ztops[ityp  ]=z;
                    //ztops[ityp+4]=z;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i=0; i<nChan; i++){
        //ztops[i]=zmin;
        if(bOccl>0){ if( ztop-ztops[i] < 0.02 ){ MultMap[iG*nChan+i] = ztops[i]; } else { MultMap[iG*nChan+i] = zmin; }; }
        else       { MultMap[iG*nChan+i] = ztops[i]; }
    }
}

__kernel void evalMultiMapSpheresElements(
    int nAtoms,
    __global float4*    atoms,
    __global float4*    coefs,
    __global int*       elemChan,
    __global float4*    poss,
    __global float*     MultMap,
    float Rpp,
    float zmin,
    float4 rotA,
    float4 rotB,
    float4 rotC,
    int bOccl,
    int nChan
){
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    __local int LELEMCHAN[32];
    float ztops[20];

    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;

    float ztop = zmin;
    for (int i=0; i<nChan; i++){
        ztops[i]=zmin;
    }

    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        LELEMCHAN[iL] = elemChan[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp    = pos - LATOMS[j].xyz;
                float3 abc   = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                float  Rvdw  = LCOEFS[j].w + Rpp;
                float r2xy   =  dot(abc.xy,abc.xy);
                float  z     = -abc.z + sqrt( Rvdw*Rvdw - r2xy );

                int ityp = LELEMCHAN[j];
                if(z>ztop){
                    ztop=z;
                }
                if(z>ztops[ityp]){
                    ztops[ityp] = z;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i=0; i<nChan; i++){
        if(bOccl>0){ if( ztop-ztops[i] < 0.02 ){ MultMap[iG*nChan+i] = ztops[i]; } else { MultMap[iG*nChan+i] = zmin; }; }
        else       { MultMap[iG*nChan+i] = ztops[i]; }
    }
}

__kernel void evalSpheresType(
    int nAtoms,
    int nType,
    __global float4*    atoms,
    __global int*       itypes,
    __global float4*    coefs,
    __global float4*    poss,
    __global float*     MultMap,
    float Rpp,
    float zmin,
    float4 rotA,
    float4 rotB,
    float4 rotC,
    int bOccl
){
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    float ztops[20];

    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);
    float3 pos = poss[iG].xyz;
    float mask = 1.0;
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf( " xyzq (%g,%g,%g,%g) coef (%g,%g,%g,%g) \n", atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w,   coefs[i].x,coefs[i].y,coefs[i].z,coefs[i].w );  } }
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf("itypes[%i] = %i \n" , i, itypes[i] ); } }

    float ztop = zmin;
    for (int i=0; i<nType; i++){
        ztops[i]=zmin;
    }

    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp    = pos - LATOMS[j].xyz;
                float3 abc   = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                float  Rvdw  = LCOEFS[j].w + Rpp;
                float r2xy   =  dot(abc.xy,abc.xy);
                float  z     = -abc.z + sqrt( Rvdw*Rvdw - r2xy );

                int ityp = itypes[j+i0];
                //if( iG==0 ){ printf( " %i : %i ityp %i \n", j+i0, itypes[j+i0], ityp ); }
                //ityp = 0;
                if(z>ztop){
                    ztop=z;
                }
                if(ityp>=0){
                    if(z>ztops[ityp]){
                        ztops[ityp]=z;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i=0; i<nType; i++){
        if(bOccl>0){ if( ztop-ztops[i] < 0.02 ){ MultMap[iG*nType+i] = ztops[i]; } else { MultMap[iG*nType+i] = zmin; }; }
        else       { MultMap[iG*nType+i] = ztops[i]; }
    }
}

__kernel void evalBondEllipses(
    int nBonds,
    __global float8*    bondPoints,
    __global float4*    poss,
    __global float*     out,
    __constant float*   Rfunc,     // definition of radial function
    float drStep,
    float Rmax,
    float elipticity,
    float zmin,
    float4 rotA,
    float4 rotB,
    float4 rotC
){
    #define  nFuncMax    128
    __local float  RFUNC [nFuncMax];
    __local float8 LATOMS[32];

    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    //float3 pos = poss[iG].xyz;
    float3 pos = rotMat ( poss[iG].xyz,  rotA.xyz, rotB.xyz, rotC.xyz );
    float invStep = 1/drStep;
    //if( iG==0 ){ for(int i=0; i<nBonds; i++){ printf( " a1(%g,%g,%g,%g) a2(%g,%g,%g,%g) \n", bondPoints[i].lo.x,bondPoints[i].lo.y,bondPoints[i].lo.z,bondPoints[i].lo.w,   bondPoints[i].hi.x,bondPoints[i].hi.y,bondPoints[i].hi.z,bondPoints[i].hi.w );  } }
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf("itypes[%i] = %i \n" , i, itypes[i] ); } }

    // local copy of Rfunc
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i0=0; i0<nFuncMax; i0+= nL ){ int i = i0+iL; RFUNC[i] = Rfunc[i]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    float R2max = Rmax*Rmax;
    float ftot=0;
    //nBonds = 1;
    for (int i0=0; i0<nBonds; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = bondPoints[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nBonds ){
                float3 p1 = rotMat ( LATOMS[j].lo.xyz,  rotA.xyz, rotB.xyz, rotC.xyz );
                float3 p2 = rotMat ( LATOMS[j].hi.xyz,  rotA.xyz, rotB.xyz, rotC.xyz );
                float2 hbond   = p2.xy - p1.xy;
                float  rbond  = length(hbond);
                hbond /= rbond;
                float2 d  = pos.xy - p1.xy;
                float  c  = dot(d,hbond);
                float2 dT = d   - hbond*c;
                //float dl = ( fabs(c) + fabs(rbond-c) - rbond )*0.5;
                //float r2 = dot(dT,dT) + dl*dl;
                //float r2 = dot(dT,dT)* max(0.0, 4.0*c*(rbond-c)/(rbond*rbond)   );
                float c_ = (c-0.5*rbond)*elipticity;
                float r2 = dot(dT,dT) + c_*c_ ;
                float r  = sqrt(r2);
                //float z       = reciprocalInterp( p1.z, p2.z, r2xy1, r2xy2  );
                float z =  mix ( p1.z, p2.z, c/rbond ) - pos.z;
                if( (r < Rmax) && (z>zmin) ){
                    float fi   = spline_Hermite_y_arr( r*invStep, RFUNC );
                    float zfunc = max( (zmin-z)/zmin, 0.0f );
                    fi*=zfunc;
                    //if( iG==0 ){ printf( " i0 %i j %i   fi %g   r2xy %g  beta %g   dp1(%g,%g,%g), dp1(%g,%g,%g) \n", i0, j,  fi, r2xy,    beta,   dp1.x,dp1.y,dp1.z, dp2.x,dp2.y,dp2.z ); }
                    //if( iG==0 ){ printf( " i0 %i j %i   z,zmin(%g,%g) zfunc %g    fi %g  fi*zfunc %g  rxy %g   sx %g  \n", i0, j,  z, zmin,  zfunc,  fi, zfunc*fi, r, r*invStep ); }
                    //ftot += fi  ;
                    ftot = fmax( fi, ftot );
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out[iG] = ftot;
}


__kernel void evalAtomRfunc(
    int nAtoms,
    __global float4*    atoms,
    __global float4*    coefs,
    __global float4*    poss,
    __global float*     FE,
    __constant float*   Rfunc,     // definition of radial function
    float drStep,
    float Rmax,
    float zmin,
    float4 rotA,
    float4 rotB,
    float4 rotC
){
    #define  nFuncMax    128
    __local float  RFUNC [nFuncMax];
    __local float4 LATOMS[32];
    __local float4 LCOEFS[32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    float3 pos = poss[iG].xyz;
    float invStep = 1/drStep;

    // local copy of Rfunc
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i0=0; i0<nFuncMax; i0+= nL ){ int i = i0+iL; RFUNC[i] = Rfunc[i]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    float ftot = 0;
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){
                float3 dp    = pos - LATOMS[j].xyz;
                float3 abc   = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                float  RvdW  = LCOEFS[j].w;
                float r2xy   =  dot(abc.xy,abc.xy);
                float r      = sqrt(r2xy);
                float  z     = -dp.z + RvdW - 1.4;  // subtract hydrogen ?
                if( (r < Rmax) && (z>zmin) ){
                    //float fi = exp( beta*rxy );
                    float fi   = spline_Hermite_y_arr( r*invStep, RFUNC );
                    float zfunc = max( (zmin-z)/zmin, 0.0f );
                    fi*=zfunc;
                    //if( iG==0 ){ printf( " i0 %i j %i   fi %g   r2xy %g  beta %g   dp1(%g,%g,%g), dp1(%g,%g,%g) \n", i0, j,  fi, r2xy,    beta,   dp1.x,dp1.y,dp1.z, dp2.x,dp2.y,dp2.z ); }
                    //if( iG==0 ){ printf( " i0 %i j %i   z,zmin(%g,%g) zfunc %g    fi %g  fi*zfunc %g  r %g   sx %g  \n", i0, j,  z, zmin,  zfunc,  fi, zfunc*fi, r, r*invStep ); }
                    //ftot += fi  ;
                    ftot = fmax( fi, ftot );
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = ftot;
}

// Get z component of electric field as the negative gradient of a Hartree potential in rotated coordinates.
__kernel void evalHartreeGradientZ(
    __global float*  pot,   // Hartree potential
    __global float4* poss,  // Position grid
    __global float*  field, // Output electric field
    int4   nGrid_H,         // Size of Hartree potential grid
    float4 T_A,             // Rows of the transformation matrix for Hartree potential grid lattice coordinates
    float4 T_B,
    float4 T_C,
    float4 origin_H,        // Real-space origin of Hartree potential grid
    float4 rotA,            // Rows of rotation matrix
    float4 rotB,
    float4 rotC,
    float4 rot_center,      // Point around which rotation is performed
    float  h                // Finite-difference step
) {

    // Get positions
    int ind = get_global_id(0);
    float3 pos = poss[ind].xyz;
    float3 d_pos = pos - rot_center.xyz;
    pos = rot_center.xyz + (float3)(dot(rotA.xyz, d_pos), dot(rotB.xyz, d_pos), dot(rotC.xyz, d_pos));
    float3 dh = (float3)(0, 0, h);
    float3 dh_rot = (float3)(dot(rotA.xyz, dh), dot(rotB.xyz, dh), dot(rotC.xyz, dh));
    float3 pos_p = pos + dh_rot;
    float3 pos_n = pos - dh_rot;

    // Interpolate potential at points around the center
    float pot_p = linearInterpB(pos_p.xyz, origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);
    float pot_n = linearInterpB(pos_n.xyz, origin_H.xyz, T_A.xyz, T_B.xyz, T_C.xyz, nGrid_H.xyz, pot);

    // Compute value of field as centered difference.
    field[ind] = 0.5*(pot_n - pot_p) / h;

}
