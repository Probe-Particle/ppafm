
#ifndef  Camera_h
#define  Camera_h

#include <math.h>
#include <cstdlib>
#include <stdint.h>

#include "fastmath.h"
#include "Vec3.h"
#include "Mat3.h"
#include "quaternion.h"

class Camera{ public:
    Vec3f  pos    = (Vec3f){0.0f,0.0f,-50.0f};
    Mat3f  rot    = Mat3fIdentity;
    float  zoom   = 10.0f;
    float  aspect = 1.0;
    float  zmin   = 10.0;
    float  zmax   = 10000.0;

    inline void lookAt( Vec3f p, float R ){ pos = p + rot.c*-R; }
    inline void lookAt( Vec3d p, float R ){ Vec3f p_; convert(p,p_); lookAt(p_,R); }

    inline float getTgX()const{ return 1.0/(zoom*aspect); }
    inline float getTgY()const{ return 1.0/(zoom);            }

    inline void word2screenOrtho( const Vec3f& pWord, Vec3f& pScreen ) const {
        Vec3f p; p.set_sub(pWord,pos);
        rot.dot_to( p, p );
        pScreen.x = p.x/(2*zoom*aspect);
        pScreen.y = p.y/(2*zoom);
        pScreen.z = (p.z-zmin)/(zmax-zmin);
    }

    inline Vec2f word2pixOrtho( const Vec3f& pWord, const Vec2f& resolution ) const {
        Vec3f p; p.set_sub(pWord,pos);
        rot.dot_to( p, p );
        return Vec2f{ resolution.x*(0.5f+p.x/(2*zoom*aspect)),
                      resolution.y*(0.5f+p.y/(2*zoom)) };
    }

    inline void word2screenPersp( const Vec3f& pWord, Vec3f& pScreen ) const {
        Vec3f p; p.set_sub(pWord,pos);
        rot.dot_to( p, p );
        float resc = zmin/(2*p.z*zoom);
        pScreen.x = p.x*resc/aspect;
        pScreen.y = p.y*resc;
        //pScreen.z = p.z/zmin;        // cz
        /*
        (2*zmin)/w      0       0              0            0
        0          (2*zmin)/h   0              0            0
        0               0       (zmax+zmin)/(zmax-zmin)    (2*zmin*zmax)/(zmax-zmin)
        0               0       0               0           -1
        //------
        x_  =  ((2*zmin)/w)  * x
        y_  =  ((2*zmin)/h ) * y
        z_  =   (zmax+zmin)/(zmax-zmin)
        */
    }

    inline Vec2f word2pixPersp( const Vec3f& pWord, const Vec2f& resolution ) const {
        Vec3f p; p.set_sub(pWord,pos);
        rot.dot_to( p, p );
        float resc = zmin/(2*p.z*zoom);
        return (Vec2f){
            resolution.x*( 0.5 + p.x*resc/aspect ),
            resolution.y*( 0.5 + p.y*resc        ) };
    }

    inline void pix2rayOrtho( const Vec2f& pix, Vec3f& ro ) const {
        float resc = 1/zoom;
        ro = rot.a*(pix.a*resc) + rot.b*(pix.b*resc);
    }

    inline Vec3f pix2rayPersp( const Vec2f& pix, Vec3f& rd ) const {
        float resc = 1/zoom;
        rd = rot.a*(pix.a*resc) + rot.b*(pix.b*resc);
    }

    inline bool pointInFrustrum( Vec3f p ) const {
        p.sub(pos);
        Vec3f c;
        rot.dot_to( p, c );
        float tgx = c.x*zoom*aspect;
        float tgy = c.y*zoom;
        float cz  = c.z*zmin;
        return (tgx>-cz)&&(tgx<cz) && (tgy>-cz)&&(tgy<cz) && (c.z>zmin)&&(c.z<zmax);
    }

    inline bool sphereInFrustrum( Vec3f p, float R ) const {
        p.sub(pos);
        Vec3f c;
        rot.dot_to( p, c );
        float my = c.z*zmin/zoom;
        float mx = my/aspect + R;  my+=R;
        return (c.x>-mx)&&(c.x<mx) && (c.y>-my)&&(c.y<my) && ((c.z+R)>zmin)&&((c.z-R)<zmax);
    }

};

#endif

