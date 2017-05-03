
// https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/sampler_t.html

__constant sampler_t sampler_1 = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

float3 tipForce( float3 dpos, float4 stiffness, float4 dpos0 ){
    float r = sqrt( dot( dpos,dpos) );
    return  (dpos-dpos0.xyz) *   stiffness.xyz              // harmonic 3D
         + dpos * ( stiffness.w *(r-dpos0.w)/r );  // radial
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


__kernel void getFEinStrokes(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 dTip,
    int nz
){
    float3 pos    =  points[get_global_id(0)].xyz; 
    for(int iz=0; iz<nz; iz++){
        float4 fe;
        //printf( " %li %i (%f,%f,%f) \n", get_global_id(0), iz, pos.x, pos.y, pos.z );
        FEs[get_global_id(0)*nz + iz]    = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
        pos    += dTip.xyz;
    }
}

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
        for(int i=0; i<100; i++){
           fe        = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );
           float3 f  = fe.xyz;
           f        += tipForce( pos-tipPos, stiffness, dpos0 );
           vel      *=       relax_params.y;       
           vel      += f   * relax_params.y;
           pos.xyz  += vel * relax_params.x;
        }
        FEs[get_global_id(0)*nz + iz] = fe;
        tipPos += dTip.xyz;
        pos    += dTip.xyz;
    }
}



