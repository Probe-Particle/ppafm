
#ifndef DynamicOpt_h
#define DynamicOpt_h

#include <cstddef>
#include <math.h>

typedef void (*ForceFunction)( int n, double * xs, double * dfs );

class DynamicOpt{
	public:
	// variables
	int n=0;
	double * pos       = NULL;
	double * vel       = NULL;
	double * force     = NULL;
	double * invMasses = NULL;

	// parameters
	double dt           = 0.05d;
	double damping      = 0.1d;

	double f_limit      = 10.0;
	double v_limit      = 1.0;
    double fscale_safe  = 1;
	double ff_safety    = 1e-32;

	// FIRE
	int    minLastNeg   = 5;
	double finc         = 1.1d;
	double fdec         = 0.5d;
	double falpha       = 0.98d;
	double kickStart    = 1.0d;

	double dt_max       = dt;
	double dt_min       = 0.1 * dt;
	double damp_max     = damping;


	int    lastNeg      = 0;

	// other
	int method    = 2;
	int stepsDone = 0;
	double t      = 0.0d;

	ForceFunction getForce = NULL;

	// ==== function declarations

	void   move_LeapFrog( double dt_loc);
	void   move_LeapFrog_vlimit();
	void   move_MDquench();
	void   move_FIRE();
	double optStep();
	bool   optimize( double convF, int nMaxSteps );

	double getFmaxAbs( );
	double getFsqSum( );

	// ==== inline functions

	inline void bindArrays( int n_, double * pos_, double * vel_, double * force_, double * invMasses_ ){
		n = n_; pos   = pos_; vel   = vel_;  force = force_; invMasses=invMasses_;
		if(pos==NULL)    pos   = new double[n];
		if(vel==NULL)    vel   = new double[n];
		if(force==NULL)  force = new double[n];
		if(invMasses==NULL) { invMasses = new double[n]; for(int i=0;i<n;i++){invMasses[i]=1.0;}  }
	}

	inline void allocate( int n_ ){
		n = n_;
		pos   = new double[n];
		vel   = new double[n];
		force = new double[n];
		invMasses = new double[n];
	}

	inline void deallocate( ){
		delete pos;
		delete vel;
		delete force;
	}

	inline void cleanForce( ){  for(int i=0; i<n; i++){ force[i]=0; } }
	inline void cleanVel  ( ){  for(int i=0; i<n; i++){ vel  [i]=0; } }

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
        vel[i] += invMasses[i]*dt_loc*force[i];
		pos[i] += dt_loc*vel[i];
	}
	stepsDone++;
	t += dt_loc;
}

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

void DynamicOpt::move_MDquench(){
	double cdamp = 1 - damping;
	double dtv = dt*fscale_safe;
	for ( int i=0; i<n; i++ ){
		vel[i]  = cdamp*vel[i] + dtv*force[i]*invMasses[i];
		pos[i] += dt*vel[i];
	}
	stepsDone++;
	t += dt;
}

void DynamicOpt::move_FIRE(){
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

	double dt_=dt;
    if( ff > f_limit*f_limit ){
        double f = sqrt(ff);
        dt_*=sqrt(f_limit/f);
        printf( "force too: %g => large limit dt: %g \n", f, dt_ );
    };
    move_LeapFrog( dt_ );
	//move_LeapFrog();
	//move_LeapFrog_vlimit();  // this does not seem to help

	//printf( " %i f v vf  %f %f %f   dt damp  %f %f \n",  stepsDone,   sqrt(ff), sqrt(vv), vf/sqrt(vv*ff),   dt_var, damp_var  );
	//stepsDone++;
}

double DynamicOpt::optStep(){
	//cleanForce( );
	getForce( n, pos, force );
	switch( method ){
		case 0: move_LeapFrog(dt);
		case 1: move_MDquench();
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
