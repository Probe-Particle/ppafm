
#ifndef  Draw3D_h
#define  Draw3D_h

//#include "fastmath.h"
#include "fastmath_light.h"
#include "integerOps.h"
#include "Vec3.h"
#include "Mat3.h"
#include "quaternion.h"


//#include "Mesh.h"

namespace Draw3D{

//typedef Vec3f (*UVfunc)(Vec2f p);

// ==== function declarations

// ========== Float Versions

void point     ( const Vec3f& vec                   );
void pointCross( const Vec3f& vec, float  sz        );
void vecInPos  ( const Vec3f& v,   const Vec3f& pos );
void line      ( const Vec3f& p1,  const Vec3f& p2  );
void arrow     ( const Vec3f& p1,  const Vec3f& p2, float sz );

void scale     ( const Vec3f& p1,  const Vec3f& p2, const Vec3f& a, float tick, float sza, float szb );

void triangle ( const Vec3f& p1,  const Vec3f& p2, const Vec3f& p3 );

void matInPos ( const Mat3f& mat, const Vec3f& pos );

void shape    ( const Vec3f& pos, const Mat3f&  rot,  int ishape, bool transposed = false );
void shape    ( const Vec3f& pos, const Quat4f& qrot, int ishape );
void shape    ( const Vec3f& pos, const Quat4f& qrot, const Vec3f& scale, int ishape );

int  cylinderStrip     ( int n, float r1, float r2, const Vec3f& base, const Vec3f& tip );
int  cylinderStrip_wire( int n, float r1, float r2, const Vec3f& base, const Vec3f& tip );
int  sphereTriangle    ( int n, float r, const Vec3f& pos, const Vec3f& a, const Vec3f& b, const Vec3f& c );
int  sphereTriangle_wire( int n, float r, const Vec3f& pos, const Vec3f& a, const Vec3f& b, const Vec3f& c );

int  circleAxis     ( int n, const Vec3f& pos, const Vec3f& v0, const Vec3f& uaxis, float R, float dca, float dsa );
int  circleAxis     ( int n, const Vec3f& pos, const Vec3f& v0, const Vec3f& uaxis, float R );

int  coneFan        ( int n, float r,                const Vec3f& base, const Vec3f& tip );
int  cone           ( int n, float phi1, float phi2, float r1, float r2, const Vec3f& base, const Vec3f& tip, bool smooth );

int  sphereOctLines ( int n, float R, const Vec3f& pos );
int  sphere_oct     ( int n, float R, const Vec3f& pos, bool wire=false );
int  capsula        ( Vec3f p0, Vec3f p1,  float r1, float r2, float theta1, float theta2, float dTheta, int nPhi, bool capped );

void kite           ( const Vec3f& pos, const Mat3f& rot, float sz );
void panel          ( const Vec3f& pos, const Mat3f& rot, const Vec2f& sz );

void text  ( const char * str, const Vec3f& pos, int fontTex, float textSize, int iend );
void text3D( const char * str, const Vec3f& pos, const Vec3f& fw, const Vec3f& up, int fontTex, float textSize, int iend );

void box( float x0, float x1, float y0, float y1, float z0, float z1, float r, float g, float b );
void bbox        ( const Vec3f& p0, const Vec3f& p1 );
void triclinicBox( const Mat3f& lvec_, const Vec3f& c0_, const Vec3f& c1_ );
void triclinicBoxT( const Mat3f& lvec_, const Vec3f& c0_, const Vec3f& c1_ );

void axis( float sc );

// ========== Double Versions

inline void point     ( const Vec3d& vec                   ){point((Vec3f)vec); }
//inline void vec       ( const Vec3d& vec                   ){vec  ((Vec3f)vec); }
inline void pointCross( const Vec3d& vec, double sz        ){pointCross((Vec3f)vec,sz); }
inline void vecInPos  ( const Vec3d& v,   const Vec3d& pos ){vecInPos((Vec3f)v,(Vec3f)pos); }
inline void line      ( const Vec3d& p1,  const Vec3d& p2  ){line ((Vec3f)p1,(Vec3f)p2); }
inline void arrow     ( const Vec3d& p1,  const Vec3d& p2, float sz  ){arrow((Vec3f)p1,(Vec3f)p2, sz); }

inline void scale     ( const Vec3d& p1,  const Vec3d& p2, const Vec3d& a, double tick, double sza, double szb ){  scale( (Vec3f)p1, (Vec3f)p2, (Vec3f)a, tick,sza,szb); };

inline void triangle ( const Vec3d& p1,  const Vec3d& p2, const Vec3d& p3 ){ triangle( (Vec3f)p1, (Vec3f)p2, (Vec3f)p3 ); };

inline void matInPos ( const Mat3d& mat, const Vec3d& pos ){  matInPos( (Mat3f)mat, (Vec3f)pos ); };

inline void shape    ( const Vec3d& pos, const Mat3d&  rot,  int ishape, bool transposed = false ){ shape( (Vec3f)pos, (Mat3f)rot, ishape, transposed ); };
inline void shape    ( const Vec3d& pos, const Quat4d& qrot, int ishape ){ shape( (Vec3f)pos, (Quat4f)qrot, ishape); };
inline void shape    ( const Vec3d& pos, const Quat4d& qrot, Vec3d& scale, int ishape ){ shape( (Vec3f)pos, (Quat4f)qrot, (Vec3f)scale, ishape); };

inline int  circleAxis     ( int n, const Vec3d& pos, const Vec3d& v0, const Vec3d& uaxis, double R ){ return circleAxis( n, (Vec3f)pos, (Vec3f)v0, (Vec3f)uaxis, R ); };
inline int  sphereOctLines ( int n, double R, const Vec3d& pos ){ return sphereOctLines ( n, R, (Vec3f)pos ); };
inline int  sphere_oct     ( int n, double R, const Vec3d& pos ){ return sphere_oct( n, R, (Vec3f)pos ); };

inline void kite     ( const Vec3d& pos, const Mat3d& rot, double sz       ){ kite ( (Vec3f)pos, (Mat3f)rot, sz ); };
inline void panel    ( const Vec3d& pos, const Mat3d& rot, const Vec2d& sz ){ panel( (Vec3f)pos, (Mat3f)rot, (Vec2f)sz ); };

inline void text     ( const char * str, const Vec3d& pos, int fontTex, float textSize, int iend ){ text(str, (Vec3f)pos, fontTex, textSize,iend); };

inline void bbox         ( const Vec3d& p0, const Vec3d& p1 )                      { bbox       ( (Vec3f)p0, (Vec3f)p1 ); };
inline void triclinicBox ( const Mat3d& lvec_, const Vec3d& c0_, const Vec3d& c1_ ){ triclinicBox( (Mat3f)lvec_, (Vec3f)c0_, (Vec3f) c1_ ); };
inline void triclinicBoxT( const Mat3d& lvec_, const Vec3d& c0_, const Vec3d& c1_ ){ triclinicBoxT( (Mat3f)lvec_, (Vec3f)c0_, (Vec3f) c1_ ); };

// ========== Arrays // Not easy to convert

void planarPolygon( int n, const int * inds, const Vec3d * points );
void polygonNormal( int n, const int * inds, const Vec3d * points );
void polygonBorder( int n, const int * inds, const Vec3d * points );

//void planarPolygon( int ipl, Mesh& mesh );
//void polygonBorder( int ipl, Mesh& mesh );
//void polygonNormal( int ipl, Mesh& mesh );

void polyLine( int n, Vec3d * ps, bool closed=false );

void points         ( int n, const Vec3d * points, float sz );
void lines          ( int nlinks, const int * links, const Vec3d * points );
void triangles      ( int nlinks, const int * links, const Vec3d * points );
void polygons       ( int nlinks, const int * ns,    const int * links, const Vec3d * points );

void vectorArray(int n, Vec3d* ps, Vec3d* vs, double sc );
void scalarArray(int n, Vec3d* ps, double* vs, double vmin, double vmax );

void simplexGrid( int na, int nb, const Vec2d& da, const Vec2d& db,  const double * hs, const double * clrs, int ncolors, const uint32_t * colorscale );
void simplexGridLines( int na, int nb, const Vec2d& da, const Vec2d& db,  const double * hs );
void simplexGridLinesToned( int na, int nb, const Vec2d& da, const Vec2d& db,  const double * hs );
void rectGridLines( Vec2i n, const Vec3d& p0, const Vec3d& da, const Vec3d& db );

//int mesh( const Mesh& mesh  );

void text( const char * str, const Vec3d& pos, int fontTex, float textSize, int istart, int iend );
void text3D( const char * str, const Vec3d& pos, int fontTex, float textSize, int iend );

void curve    ( float tmin, float tmax,   int n, Func1d3 func );

inline void colorScale( int n, Vec3d pos, Vec3d dir, Vec3d up, void (_colorFunc_)(float f) );

void rigidTransform( const Vec3f& pos, Mat3f rot,          const Vec3f& sc, bool trasposed = false );
void rigidTransform( const Vec3f& pos, const Quat4f& qrot, const Vec3f& sc, bool trasposed = false );
void rigidTransform( const Vec3d& pos, const Mat3d& rot,   const Vec3d& sc, bool trasposed = false );
void rigidTransform( const Vec3d& pos, const Quat4d& qrot, const Vec3d& sc, bool trasposed = false );

// ==== inline functions

// TODO : This could be perhaps moved to Camera

inline void toGLMat( const Vec3f& pos, const Mat3f& rot, float* glMat ){
    //printf("pos (%3.3f,%3.3f,%3.3f)\n", pos.x,pos.y,pos.z);
    glMat[0 ] = rot.ax;   glMat[1 ] = rot.ay;   glMat[2 ] = rot.az;   glMat[3 ]  = 0;
    glMat[4 ] = rot.bx;   glMat[5 ] = rot.by;   glMat[6 ] = rot.bz;   glMat[7 ]  = 0;
    glMat[8 ] = rot.cx;   glMat[9 ] = rot.cy;   glMat[10] = rot.cz;   glMat[11]  = 0;
    glMat[12] = pos. x;   glMat[13] = pos. y;   glMat[14] = pos. z;   glMat[15]  = 1;
};

inline void toGLMatT( const Vec3f& pos, const Mat3f& rot, float* glMat ){
    //printf("pos (%3.3f,%3.3f,%3.3f)\n", pos.x,pos.y,pos.z);
    glMat[0 ] = rot.ax;   glMat[1 ] = rot.bx;   glMat[2 ] = rot.cx;   glMat[3 ]  = 0;
    glMat[4 ] = rot.ay;   glMat[5 ] = rot.by;   glMat[6 ] = rot.cy;   glMat[7 ]  = 0;
    glMat[8 ] = rot.az;   glMat[9 ] = rot.bz;   glMat[10] = rot.cz;   glMat[11]  = 0;
    glMat[12] = pos. x;   glMat[13] = pos. y;   glMat[14] = pos. z;   glMat[15]  = 1;
};

inline void toGLMatCam( const Vec3f& pos, const Mat3f& rot, float* glMat ){
	glMat[0 ] = rot.ax;   glMat[1 ] = rot.bx;   glMat[2 ] = -rot.cx;   glMat[3 ]  = 0;
	glMat[4 ] = rot.ay;   glMat[5 ] = rot.by;   glMat[6 ] = -rot.cy;   glMat[7 ]  = 0;
	glMat[8 ] = rot.az;   glMat[9 ] = rot.bz;   glMat[10] = -rot.cz;   glMat[11]  = 0;
	glMat[12] = -pos. x;  glMat[13] = -pos. y;  glMat[14] = -pos. z;   glMat[15]  = 1;
};

inline void toGLMat( const Vec3f& pos, Mat3f rot, const Vec3f& sc, float* glMat ){
    rot.mul(sc);
    //rot.print();
    toGLMat( pos, rot, glMat );
};

inline void toGLMatCam( const Vec3f& pos, Mat3f rot, const Vec3f& sc, float* glMat ){
    rot.divT(sc);
    toGLMatCam( pos, rot, glMat );
};

inline void toGLMat   ( const Vec3f& pos, const Quat4f& qrot, float* glMat ){ toGLMat( pos, qrot.toMat(), glMat );    };
inline void toGLMatCam( const Vec3f& pos, const Quat4f& qrot, float* glMat ){ toGLMatCam( pos, qrot.toMat(), glMat ); };

inline void toGLMat   ( const Vec3f& pos, const Quat4f& qrot, const Vec3f& sc, float* glMat ){ toGLMat( pos, qrot.toMat(), sc, glMat );    };
inline void toGLMatCam( const Vec3f& pos, const Quat4f& qrot, const Vec3f& sc, float* glMat ){ toGLMatCam( pos, qrot.toMat(), sc, glMat ); };

inline void toGLMat       ( const Vec3d& pos, const Mat3d& rot,   float* glMat   ){ toGLMat   ( (Vec3f)pos, (Mat3f)rot, glMat ); };
inline void toGLMatT      ( const Vec3d& pos, const Mat3d& rot,   float* glMat   ){ toGLMatT  ( (Vec3f)pos, (Mat3f)rot, glMat ); };
inline void toGLMatCam    ( const Vec3d& pos, const Mat3d& rot,   float* glMat   ){ toGLMatCam( (Vec3f)pos, (Mat3f)rot, glMat ); };
inline void toGLMat       ( const Vec3d& pos, const Quat4d& qrot, float* glMat   ){ toGLMat   ( (Vec3f)pos, ((Quat4f)qrot).toMat(), glMat ); };
inline void toGLMatCam    ( const Vec3d& pos, const Quat4d& qrot, float* glMat   ){ toGLMatCam( (Vec3f)pos, ((Quat4f)qrot).toMat(), glMat ); };
inline void toGLMat       ( const Vec3d& pos, const Mat3d& rot,   const Vec3d& sc, float* glMat   ){ toGLMat   ( (Vec3f)pos, (Mat3f)rot, (Vec3f)sc, glMat ); };
inline void toGLMatCam    ( const Vec3d& pos, const Mat3d& rot,   const Vec3d& sc, float* glMat   ){ toGLMatCam( (Vec3f)pos, (Mat3f)rot, (Vec3f)sc, glMat ); };
inline void toGLMat       ( const Vec3d& pos, const Quat4d& qrot, const Vec3d& sc, float* glMat   ){ toGLMat   ( (Vec3f)pos, ((Quat4f)qrot).toMat(), (Vec3f)sc, glMat ); };
inline void toGLMatCam    ( const Vec3d& pos, const Quat4d& qrot, const Vec3d& sc, float* glMat   ){ toGLMatCam( (Vec3f)pos, ((Quat4f)qrot).toMat(), (Vec3f)sc, glMat ); };




}; // namespace Draw3D

#endif

