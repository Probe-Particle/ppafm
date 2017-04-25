
// https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/sampler_t.html

__constant sampler_t sampler_1 = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

float3 tipForce( float3 dpos, float4 stiffness, float4 dpos0 ){
    float r = sqrt( dot( dpos,dpos) );
    return dpos.xyz *   stiffness.xyz                 // harmonic 3D
         + dpos.xyz * ( stiffness.w *(r-dpos.w)/r );  // radial
}

float4 interpFE( float3 pos, float3 dinvC, float3 dinvC, float3 dinvC, __read_only image3d_t imgIn ){
    const float4 coord = (float4)( dot(pos,dinvA),dot(pos,dinvB),dot(pos,dinvC) );
    return read_imagef( imgIn, sampler_1, coord );
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
    __global  float *      FEs,
){
    const float4 coord     = points[get_global_id(0)];
    vals[get_global_id(0)] = read_imagef(imgIn, sampler_1, coord);
}

__kernel void relaxPoints(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float *      FEs,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params  // (dt,damp,tmin,tmax)
){
    float4 tipPos = points[get_global_id(0)];
    float3 pos    =  tipPos.xyz + dpos0.xyz; 
    float4 fe;
    float3 vel    = 0.0f;
    for(int i=0; i<1000; i++){
       fe        = read_imagef( imgIn, sampler_1, pos ); /// this would work only for unitary cell
       float3 f  = fe.xyz;
       float3 f += tipForce( pos-tipPos, stiffness, dpos0 );       
       vel      += f   * dt;
       pos.xyz  += vel * dt;
    }
    vals[get_global_id(0)] = fe;
}

__kernel void relaxStrokes(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float *      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 dTip,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    int nz
){
    float4 tipPos = points[get_global_id(0)];
    float3 pos    = tipPos.xyz + dpos0.xyz; 
    for(int iz=0; iz<nz; iz++){
        float4 fe;
        float3 vel   = 0.0f;
        for(int i=0; i<100; i++){
           fe        = interpFE( , sampler_1, imgIn );
           float3 f  = fe.xyz;
           float3 f += tipForce( pos-tipPos, stiffness, dpos0 );       
           vel      += f   * dt;
           pos.xyz  += vel * dt;
        }
        vals[get_global_id(0)] = fe;
        tipPos += dpos0;
        pos    += dpos0;
    }
}



