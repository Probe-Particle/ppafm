
#ifndef DynamicOpt_h
#define DynamicOpt_h

#include <cstddef>
#include <math.h>
#include "VecN.h"

typedef void (*ForceFunction)( int n, double * xs, double * dfs );

class DynamicOpt{ public:

    //static int iDebug = 0;

    double ff=0,vv=0,vf=0;

    // variables
    int n=0;
    double * pos       = 0;
    double * vel       = 0;
    double * force     = 0;
    double * invMasses = 0;

    // parameters
    double dt           = 0.05;
    double damping      = 0.1;

    double f_limit      = 10.0;
    //double v_limit      = 1.0;
    double l_limit      = 0.2;
    //double fscale_safe  = 1;
    double scale_dt  = 1;
    double ff_safety    = 1e-32;

    // FIRE
    int    minLastNeg   = 5;
    double finc         = 1.1;
    double fdec         = 0.5;
    double falpha       = 0.98;
    double kickStart    = 5.0;

    double dt_max       = dt;
    double dt_min       = 0.1 * dt;
    double damp_max     = damping;

    int    lastNeg      = 0;

    // other
    int method    = 2;
    int stepsDone = 0;
    double t      = 0.0;

    ForceFunction getForce = NULL;

    // ==== function declarations

    void   move_LeapFrog( double dt_loc );
    //void   move_LeapFrog_vlimit();
    double move_GD      ( double dt_loc );
    double move_GD_safe ( double dt_loc );
    void   move_MD      ( double dt_loc );
    double move_MD_safe ( double dt_loc );
    double move_FIRE();
    double optStep();
    bool   optimize( double convF, int nMaxSteps );

    double getFmaxAbs( );
    double getFsqSum( );

    // ==== inline functions

/*
    inline void initMass(double invM){ for(int i=0;i<n;i++){invMasses[i]=invM;} };
    inline void bindArrays( int n_, double * pos_, double * vel_, double * force_, double * invMasses_ ){
        n = n_;
        if(pos_  ==0){ _realloc(pos  ,n); }else{ pos   = pos_;   };
        if(vel_  ==0){ _realloc(vel  ,n); }else{ vel   = vel_;   };
        if(force_==0){ _realloc(force,n); }else{ force = force_; };
        if(invMasses_==0) { _realloc(invMasses,n); initMass(1.0); }else{ invMasses=invMasses_; }
    }
*/

    inline void setInvMass(double invM){  if(invMasses==0){ _realloc(invMasses,n);}  for(int i=0;i<n;i++){invMasses[i]=invM;} };

    inline void bindArrays( int n_, double * pos_, double * vel_, double * force_, double * invMasses_ ){
        n = n_; pos=pos_;  vel=vel_; force=force_; invMasses=invMasses_;
    }

    inline void bindOrAlloc( int n_, double * pos_, double * vel_, double * force_, double * invMasses_ ){
        n = n_;
        if(pos_  ==0){ _realloc(pos  ,n); }else{ pos   = pos_;   };
        if(vel_  ==0){ _realloc(vel  ,n); }else{ vel   = vel_;   };
        if(force_==0){ _realloc(force,n); }else{ force = force_; };
        //if(invMasses_==0) { _realloc(invMasses,n); setInvMass(1.0); }else{ invMasses=invMasses_; }
        if(invMasses_==0) setInvMass(1.0);
    }

    inline void realloc( int n_ ){
        n = n_;
        _realloc(pos    ,n);
        _realloc(vel    ,n);
        _realloc(force  ,n);
        _realloc(invMasses,n);
    }

    inline void dealloc( ){
        _dealloc(pos);
        _dealloc(vel);
        _dealloc(force);
        _dealloc(invMasses);
    }

    inline void unbindAll(){
        n=0; pos=0; vel=0; force=0; invMasses=0;
    }

    inline void cleanForce( ){  for(int i=0; i<n; i++){ force[i]=0; } }
    inline void cleanVel  ( ){  for(int i=0; i<n; i++){ vel  [i]=0; } }

    //inline double limit_dt_x2 (double xx,double xmax){ double sc=1.0; if( xx > (xmax*xmax) ){ sc= fmin( sc, xmax/sqrt(xx) ); }; return sc;       }
    //inline double limit_dt_vf2(double ff, double vv ){ scale_dt=fmin(limit_dt_x2(ff,f_limit),limit_dt_x2(vv,v_limit));         return scale_dt; }

    inline void initOpt( double dt_, double damp_ ){
        dt      = dt_max   = dt_;  dt_min=0.1*dt_max;
        damping = damp_max = damp_;
        cleanForce( );
        cleanVel  ( );
    }

};


//////////////////////////////////
//     Implementation
//////////////////////////////////

#include <cstdio>// DEBUG

// ===============  MoveSteps

void DynamicOpt::move_LeapFrog(double dt_loc){
    //double dt_ = dt*fscale_safe;
    for ( int i=0; i<n; i++ ){
        //printf( "i %i v %g f %g p %g iM %g \n", i, vel[i],force[i],pos[i],invMasses[i]  );
        vel[i] += force[i]*invMasses[i]*dt_loc;
        pos[i] += vel[i]*dt_loc;
    }
    stepsDone++;
    t += dt_loc;
}

/*
void DynamicOpt::move_LeapFrog_vlimit(){
    double dtv = dt*fscale_safe;
    double vmax = 0.0d;
    for ( int i=0; i<n; i++ ){
        double v = vel[i] + invMasses[i]*dtv*force[i];
        vmax = fmax( fabs(v), vmax );
        vel[i]=v;
    }
    double dtp = dt;
    if( vmax>v_limit ) dtp=v_limit/vmax;
    //printf("vmax %g dtp  %g dtv %g\n", vmax);
    for ( int i=0; i<n; i++ ){
        pos[i] += dtp*vel[i];
    }
    stepsDone++;
    t += dt;
}
*/

double DynamicOpt::move_GD(double dt_loc){
    //double dt_ = dt*fscale_safe;
    double dr2 = 0;
    for ( int i=0; i<n; i++ ){
        double dri = force[i]*dt_loc;
        pos[i] += dri; dr2+=dri*dri;
        //printf(  "move_GD[%i] p %g f %g \n", i, pos[i], force[i] );
        //pos[i] += force[i]*dt_loc;
    }
    if(iDebug>0) printf( " move_GD dt %g dr %g \n", dt_loc, sqrt(dr2) );
    stepsDone++;
    t += dt_loc;
    return dr2;
}

double DynamicOpt::move_GD_safe(double dt_loc){
    double fmax = VecN::absmax(n,force);
    scale_dt = fmin( 1, f_limit/fmax );
    dt_loc*=scale_dt;
    move_GD(dt_loc);
    stepsDone++;
    t += dt_loc;
    return fmax;
}

void DynamicOpt::move_MD(double dt_loc){
    double cdamp = 1 - damping;
    for ( int i=0; i<n; i++ ){
        vel[i]  = cdamp*vel[i] + force[i]*invMasses[i]*dt_loc;
        pos[i] += vel[i]*dt_loc;
    }
    stepsDone++;
    t += dt;
}

/*
double DynamicOpt::move_MD_safe(double dt_loc){
    double fmax = VecN::absmax(n,force);
    //scale_dt = fmin(1,fmin( v_limit/VecN::absmax(n,vel), f_limit/fmax ));
    dt_loc*=scale_dt;
    move_MD(dt_loc);
    return fmax;
}
*/

/*
double DynamicOpt::move_FIRE(){
    double ff=0,vv=0,vf=0;
    for(int i=0; i<n; i++){
        double fi = force[i];
        double vi = vel[i];
        ff += fi*fi;
        vv += vi*vi;
        vf += vi*fi;
    }
    if( vf < 0.0 ){
        //dt       = dt * fdec;
        dt       = fmax( dt * fdec, dt_min );
        damping  = damp_max;
        lastNeg  = 0;
        //cleanVel  ( );
        for(int i=0; i<n; i++){ vel[i] = kickStart*dt*force[i]; }
        //for(int i=0; i<n; i++){ vel[i] = dmax*force[i]*sqrt(1/ff)/dt_var; }
    }else{
        double cf  =      damping * sqrt(vv/(ff+ff_safety));
        //double cf     =     damping * sqrt(vv/ff);
        double cv     = 1 - damping;
        for(int i=0; i<n; i++){
            vel[i]    = cv * vel[i]  + cf * force[i];
        }
        if( lastNeg > minLastNeg ){
            dt        = fmin( dt * finc, dt_max );
            damping   = damping  * falpha;
        }
        lastNeg++;
    }
    move_LeapFrog( dt*limit_dt_vf2(ff,vv) );
    //move_LeapFrog();
    //move_LeapFrog_vlimit();  // this does not seem to help
    //printf( " %i f v vf  %f %f %f   dt damp  %f %f \n",  stepsDone,   sqrt(ff), sqrt(vv), vf/sqrt(vv*ff),   dt_var, damp_var  );
    //stepsDone++;
    return ff;
}
*/

double DynamicOpt::move_FIRE(){
	ff=0,vv=0,vf=0;
	//printf( "DEBUG 5.5.1: %i\n", n  );
    //printf( "move_FIRE  n %i \n", n );
	for(int i=0; i<n; i++){
		double fi = force[i];
		double vi = vel[i];
		ff += fi*fi;
		vv += vi*vi;
		vf += vi*fi;
        //printf( "move_FIRE %i f %g v %g p %g \n", i, force[i], vel[i], pos[i] );
	}
	//printf( "DEBUG 5.5.2 \n" );
	if( vf < 0.0 ){
	//if( (vf<0.0)||bOverLimit){
		//dt       = dt * fdec;
		dt       = fmax( dt * fdec, dt_min );
	  	damping  = damp_max;
		lastNeg  = 0;
		cleanVel( );
		//move_GD(dt);
		//for(int i=0; i<n; i++){ vel[i] = kickStart*dt*force[i]; }
		//for(int i=0; i<n; i++){ vel[i] = force[i] * 0.5*sqrt(vv/(ff+ff_safety)); }
		//for(int i=0; i<n; i++){ vel[i] = dmax*force[i]*sqrt(1/ff)/dt_var; }
	}else{
		double cf  =      damping * sqrt(vv/(ff+ff_safety));
		//double cf     =     damping * sqrt(vv/ff);
		double cv     = 1 - damping;
		for(int i=0; i<n; i++){
			vel[i]    = cv * vel[i]  + cf * force[i];
		}
		if( lastNeg > minLastNeg ){
			dt        = fmin( dt * finc, dt_max );
			damping   = damping  * falpha;
		}
		lastNeg++;
	}
	//printf( "DEBUG 5.5.3 \n" );

	double dt_=dt;

    if( ff>(f_limit*f_limit) ){
        double f = sqrt(ff);
        //printf( "f %g f_limit %g \n ", f, f_limit );
        //if( ff>(100*f_limit*f_limit) ){ // Gradient descent for extremely high forces
        if( f>(f_limit) ){ // Gradient descent for extremely high forces
            if(iDebug>0) printf( "f(%g)>(%g) => GradinentDescent dt %g \n", f, f_limit, l_limit/f );
            cleanVel();
            move_GD( l_limit/f ); // do GD step of length == l_limit
            return ff;
        }
        dt_*=sqrt( f_limit/f );
        if(iDebug>0) printf( "force too large: %g => limit dt: %g \n", f, dt_ );
    };

    move_LeapFrog( dt_ );
	//move_LeapFrog();
	//move_LeapFrog_vlimit();  // this does not seem to help
	//printf( " %i f v vf  %f %f %f   dt damp  %f %f \n",  stepsDone,   sqrt(ff), sqrt(vv), vf/sqrt(vv*ff),   dt_var, damp_var  );
	//stepsDone++;
	return ff;
}


double DynamicOpt::optStep(){
    //cleanForce( );
    getForce( n, pos, force );
    switch( method ){
        //case 0: move_LeapFrog(dt);
        case 0: move_GD(dt);
        case 1: move_MD(dt);
        case 2: move_FIRE();
    }
    return getFmaxAbs( );
}

bool DynamicOpt::optimize( double convF, int nMaxSteps ){
    for( int i=0; i<nMaxSteps; i++ ){
        double f = optStep();
        if( f < convF ) return true;
    }
    return false;
}

// =============== common rutines

double DynamicOpt::getFmaxAbs( ){
    double fmax = 0;
    for(int i=0; i<n; i++){
        double fi = fabs( force[i] );
        fmax=(fi>fmax)?fi:fmax;
    }
    return fmax;
}

double DynamicOpt::getFsqSum( ){
    double ff = 0;
    for(int i=0; i<n; i++){
        double fi = force[i];
        ff += fi*fi;
    }
    return ff;
}


#endif
