
#define R2SAFE 1e-8

float4 getCoulomb( float4 atom, float3 pos ){
     float3  dp  =  pos - atom.xyz;
     //float  ir = 1.0/length(dp);
     float   ir2 = 1.0/( dp.x*dp.x + dp.y*dp.y + dp.z*dp.z +   R2SAFE  );
     float   ir  = sqrt(ir2);
     float   E   = atom.w*sqrt(ir2);
     return (float4)(E, dp*(E*ir2));
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
