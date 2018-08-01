
// https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/sampler_t.html

__constant sampler_t sampler_1 = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

/*
float3 tipForce( float3 dpos, float4 stiffness, float4 dpos0 ){
    float r = sqrt( dot( dpos,dpos) );
    return  (dpos-dpos0.xyz) * stiffness.xyz              // harmonic 3D
         + dpos * ( stiffness.w * (r-dpos0.w)/r );  // radial
}

*/

inline float3 rotMat( float3 v, float3 a, float3 b, float3 c ){ return (float3)(dot(v,a),dot(v,b),dot(v,c)); }
inline float3 rotMatT( float3 v,  float3 a, float3 b, float3 c  ){ return a*v.x + b*v.y + c*v.z; }

float3 tipForce( float3 dpos, float4 stiffness, float4 dpos0 ){
    float r = sqrt( dot( dpos,dpos) );
    return  (dpos-dpos0.xyz) * stiffness.xyz              // harmonic 3D
         + dpos * ( stiffness.w * (r-dpos0.w)/r );  // radial
}

float4 interpFE( float3 pos, float3 dinvA, float3 dinvB, float3 dinvC, __read_only image3d_t imgIn ){
    const float4 coord = (float4)( dot(pos,dinvA),dot(pos,dinvB),dot(pos,dinvC), 0.0f );
    return read_imagef( imgIn, sampler_1, coord );
    //return coord;
}

// this should be macro, to pass values by reference
void move_LeapFrog( float3 f, float3 p, float3 v, float2 RP ){
    v  =  f * RP.x + v*RP.y;
    p +=  v * RP.x;
}

#define FTDEC 0.5f
#define FTINC 1.1f
#define FDAMP 0.99f
#define N_RELAX_STEP_MAX  64
#define F2CONV  1e-8

// this should be macro, to pass values by reference
void move_FIRE( float3 f, float3 p, float3 v, float2 RP, float4 RP0 ){
    // RP0 = (t?,damp0,tmin,tmax)
    float ff   = dot(f,f);
    float vv   = dot(v,v);
    float vf   = dot(v,f);
    if( vf < 0 ){
        v      = 0.0f;
        RP.x   = max( RP.x*FTDEC, RP0.z );    // dt
        RP.y   = RP0.y;                      // damp
    }else{
        v      = v*(1-RP.y) + f*RP.y * sqrt(vv/ff);
        RP.x   = min( RP.x*FTINC, RP0.w );   // dt
        RP.y  *= FDAMP;                     // damp
    }
    // normal leap-frog times step
    v +=  f * RP.x;
    p +=  v * RP.x;
}

__kernel void getFEinPoints(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC
){
    //const float4 coord     = points[get_global_id(0)];
    //vals[get_global_id(0)] = read_imagef(imgIn, sampler_1, coord);
    FEs[get_global_id(0)]    = interpFE( points[get_global_id(0)].xyz, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
}

__kernel void getFEinPointsShifted(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 dpos0
){
    FEs[get_global_id(0)] = interpFE( points[get_global_id(0)].xyz+dpos0.xyz, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
}

__kernel void getFEinStrokes(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 dTip,
    float4 dpos0,
    int nz
){
    float3 pos    =  points[get_global_id(0)].xyz + dpos0.xyz; 
    for(int iz=0; iz<nz; iz++){
        float4 fe;
        //printf( " %li %i (%f,%f,%f) \n", get_global_id(0), iz, pos.x, pos.y, pos.z );
        FEs[get_global_id(0)*nz + iz]    = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
        pos    += dTip.xyz;
    }
}


__kernel void getFEinStrokesTilted(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 dTip,
    float4 dpos0,
    int nz
){
    float3 pos    =  points[get_global_id(0)].xyz + dpos0.xyz; 
    for(int iz=0; iz<nz; iz++){
        //printf( " %li %i (%f,%f,%f) \n", get_global_id(0), iz, pos.x, pos.y, pos.z );
        float4 fe   = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
        float4 fe_  = fe;
        fe_.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
        FEs[get_global_id(0)*nz + iz]    = fe_;
        pos    += dTip.xyz;
    }
}

/*
__kernel void getZisoTilted(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      zIso,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 dTip,
    float4 dpos0,
    int nz, float iso
){
    float3 pos    =  points[get_global_id(0)].xyz + dpos0.xyz; 
    float4 ofe = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
    float4 ofe = ofe_.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
    float4 fe;
    for(int iz=1; iz<nz; iz++){
        //printf( " %li %i (%f,%f,%f) \n", get_global_id(0), iz, pos.x, pos.y, pos.z );
        fe = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
        if( fe.z ) break;
        ofe      = fe;
        pos    += dTip.xyz;
    }
    FEs[ get_global_id(0)]
}
*/

__kernel void relaxPoints(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params  // (dt,damp,tmin,tmax)
){
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos    =  tipPos.xyz + dpos0.xyz; 
    float4 fe;
    float3 vel    = 0.0f;
    for(int i=0; i<1000; i++){
        fe        = read_imagef( imgIn, sampler_1, (float4)(pos,0.0f) ); /// this would work only for unitary cell
        float3 f  = fe.xyz;
        f        += tipForce( pos-tipPos, stiffness, dpos0 );    
        vel      *=       relax_params.y;   
        vel      += f   * relax_params.x;
        pos.xyz  += vel * relax_params.x;
    }
    FEs[get_global_id(0)] = fe;
}

__kernel void relaxStrokes(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 dTip,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    int nz
){
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos    = tipPos.xyz + dpos0.xyz; 
    
    //printf( " %li (%f,%f,%f)  \n",  get_global_id(0), tipPos.x, tipPos.y, tipPos.z);
    
    for(int iz=0; iz<nz; iz++){
        float4 fe;
        float3 vel   = 0.0f;
        for(int i=0; i<N_RELAX_STEP_MAX; i++){
            fe        = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
            float3 f  = fe.xyz;
            f        += tipForce( pos-tipPos, stiffness, dpos0 );
            vel      *=       relax_params.y;       
            vel      += f   * relax_params.y;
            pos.xyz  += vel * relax_params.x;
            if(dot(f,f)<F2CONV) break;
        }
        //FEs[get_global_id(0)*nz + iz] = fe;
        FEs[get_global_id(0)*nz + iz].xyz = pos;
        tipPos += dTip.xyz;
        pos    += dTip.xyz;
    }
}

__kernel void relaxStrokesTilted(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    int nz
){
    float3 dTip   = tipC.xyz * tipC.w;
    float3 tipPos = points[get_global_id(0)].xyz;

    float4 dpos0_=dpos0; dpos0_.xyz= rotMatT( dpos0_.xyz , tipA.xyz, tipB.xyz, tipC.xyz );
    float3 pos     = tipPos.xyz + dpos0_.xyz; 

    for(int iz=0; iz<nz; iz++){
        float4 fe;
        float3 vel   = 0.0f;
        for(int i=0; i<N_RELAX_STEP_MAX; i++){
            fe        = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
            float3 f  = fe.xyz;
            float3 dpos  = pos-tipPos;

            float3 dpos_ = rotMat( dpos, tipA.xyz, tipB.xyz, tipC.xyz );    // to tip-coordinates
            float3 ftip  = tipForce( dpos_, stiffness, dpos0 );
            f            += rotMatT( ftip, tipA.xyz, tipB.xyz, tipC.xyz );      // from tip-coordinates
            
            //f      +=  tipForce( dpos, stiffness, dpos0_ );  // Not rotated
            vel      *=       relax_params.y;
            vel      += f   * relax_params.y;
            pos.xyz  += vel * relax_params.x;
            if(dot(f,f)<F2CONV) break;
        }
        
        if(1){ // output tip-rotated force
            float4 fe_  = fe;
            fe_.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
            fe_.w   = fe.w;
            FEs[get_global_id(0)*nz + iz] = fe_;
        }else{ // output molecule-rotated force 
            FEs[get_global_id(0)*nz + iz] = fe;
            //FEs[get_global_id(0)*nz + iz].xyz = pos;
        }
        tipPos += dTip.xyz;
        pos    += dTip.xyz;
    }
}

__kernel void convolveZ(
    __global  float4* Fin,
    __global  float4* Fout,
    __global  float*  weighs,
    int nzin, int nzout
){
    int ioffi = get_global_id(0)*nzin;
    int ioffo = get_global_id(0)*nzout;
    for(int izo=0; izo<nzout; izo++){
        float4 fe = 0.0f;
        for(int izi=0; izi<(nzin-izo); izi++){
            fe += Fin[ ioffi + izi ] * weighs[ izi + izo ];
            //fe +=  tanh( Fin[ ioffi + izi ] ) * weighs[ izi - izo ];
        }
        //if( ioffi == 0 ){ printf( "izo %i w[i] %e \n", izo, weighs[ izo ] ); }
        Fout[ ioffo + izo ] = fe;
        //Fout[ ioffo + izo ] = weighs[ izo ];
        //Fout[ ioffo + izo ] = (float4) izo;
        //Fout[ ioffo + izo ] = Fin[ ioffi + izo ];
    }
}

__kernel void izoZ(
    __global  float4* Fin,
    __global  float*  zMap,
    int nz,   float iso
){
    int ioffi = get_global_id(0)*nz;
    float4 ofe = Fin[ ioffi ];
    for(int iz=1; iz<nz; iz++){
        float4 fe = Fin[ ioffi + iz ];
        // zMap[get_global_id(0)] = i;
        if( fe.z > iso ){
            float t = (iso - ofe.z)/(fe.z - ofe.z);
            zMap[get_global_id(0)] = iz + t;
            return;
        }
        ofe = fe;
    }
    zMap[get_global_id(0)] = -1;
}


