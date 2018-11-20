
#define R2SAFE          1e-4f
#define COULOMB_CONST   14.399644f  // [eV/e]

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
        if(i>=nAtoms) break;
        //if(iL==0) printf("%i (%f,%f,%f)  %f \n", i, atoms[i].x, atoms[i].y, atoms[i].z, atoms[i].w );
        
        LATOMS[iL] = atoms[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            fe += getCoulomb( LATOMS[j], pos );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = fe;
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
            if( (j+i0)<nAtoms ) fe += getLorenz( LATOMS[j], coefs[j], pos );
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
                float   R  = coefs[j].z;
                float   r2 = dot(abc.xy,abc.xy);
                float dxy2 = r2/( (R*R) );
                if( ( dxy2 < 1.0f ) && ( abc.z < dzmax ) ){
                    //fe += 1-dxy2;
                    //fe += ( 1-(abc.z/dzmax) ) * sqrt( 1- dxy2 );
                    fe += ( 1.0f-(abc.z/dzmax) ) * ( 1.0f- sqrt(dxy2) );
                    //fe += coefs[j].w;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = fe;
}


__kernel void evalQDisk(
    int nAtoms, 
    __global float4*    atoms,
    //__global float4*    coefs,
    __global float4*    poss,
    __global float*     FE,
    float dzmax,
    float4 rotA,
    float4 rotB,
    float4 rotC
){
    __local float4 LATOMS[32];
    //__local float4 LCOEFS[32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);
   
    float3 pos = poss[iG].xyz;
    float fe = 0.0f;
    //if( iG==0 ){ for(int i=0; i<nAtoms; i++){ printf( " xyzq (%g,%g,%g,%g) coef (%g,%g,%g,%g) \n", atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w,   coefs[i].x,coefs[i].y,coefs[i].z,coefs[i].w );  } }
    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        //LCOEFS[iL] = coefs[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){ 
                float3 dp  =  pos - LATOMS[j].xyz;
                float3 abc = (float3)( dot(dp,rotA.xyz), dot(dp,rotB.xyz), dot(dp,rotC.xyz) );
                float   R  = 1.0f-(abc.z/dzmax);
                float   r2 = dot(abc.xy,abc.xy);
                float dxy2 = r2/(R*R);
                if( (R>0.0) && (dxy2<1.0) ){
                    fe +=  LATOMS[j].w * ( 1.0f- sqrt(dxy2) );
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    FE[iG] = fe;
}



