
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <stdint.h>

#include "Draw3D.h" // THE HEADER

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

namespace Draw3D{

    constexpr int     ncolors = 5;
    static uint32_t   colors_rainbow[ncolors] = { 0xFF000000, 0xFFFF0000, 0xFF00FF00, 0xFF00FFFF, 0xFFFFFFFF };



void rigidTransform( const Vec3f& pos, Mat3f rot, const Vec3f& sc, bool trasposed ){
    float glMat[16];
    if( trasposed ){
        rot.mul(sc);
        toGLMatT(pos,rot,glMat);
    }else{
        //rot.div(sc);
        rot.mul(sc);
        toGLMat(pos,rot, glMat);
    };
    glMultMatrixf( glMat );
}
void rigidTransform( const Vec3f& pos, const Quat4f& qrot, const Vec3f& sc, bool trasposed ){ rigidTransform( pos, qrot.toMat(), sc, trasposed ); };

void rigidTransform( const Vec3d& pos, const Mat3d& rot,   const Vec3d& sc, bool trasposed ){ rigidTransform( (Vec3f)pos, (Mat3f)rot, (Vec3f)sc, trasposed ); };
void rigidTransform( const Vec3d& pos, const Quat4d& qrot, const Vec3d& sc, bool trasposed ){ rigidTransform( (Vec3f)pos, ((Quat4f)qrot).toMat(), (Vec3f)sc, trasposed ); };




// ====== From Draw.cpp

void setRGB( uint32_t i ){
	constexpr float inv255 = 1.0f/255.0f;
	//glColor3f( (i&0xFF)*inv255, ((i>>8)&0xFF)*inv255, ((i>>16)&0xFF)*inv255 );
	glColor3f( ((i>>16)&0xFF)*inv255, ((i>>8)&0xFF)*inv255, (i&0xFF)*inv255  );
}

void setRGBA( uint32_t i ){
	constexpr float inv255 = 1.0f/255.0f;
	glColor4f( (i&0xFF)*inv255, ((i>>8)&0xFF)*inv255, ((i>>16)&0xFF)*inv255, ((i>>24)&0xFF)*inv255 );
}

void color_of_hash( int i ){
	//constexpr float inv255 = 1.0f/255.0f;
	int h = hash_Wang( i );
	setRGB( h );
	//glColor3f( (h&0xFF)*inv255, ((h>>8)&0xFF)*inv255, ((h>>16)&0xFF)*inv255 );
}

void colorScale( double d, int ncol, const uint32_t * colors ){
    constexpr float inv255 = 1.0f/255.0f;
    d*=(ncol-1);
    int icol = (int)d;
    d-=icol; double md = 1-d;
    uint32_t clr1=colors[icol];
    uint32_t clr2=colors[icol];
    glColor3f(
        ( d*(  colors[icol]    &0xFF) + md*( colors[icol]     &0xFF ))*inv255,
        ( d*((colors[icol]>>8 )&0xFF) + md*((colors[icol]>>8 )&0xFF ))*inv255,
        ( d*((colors[icol]>>16)&0xFF) + md*((colors[icol]>>16)&0xFF ))*inv255
    );
}

/*
void Draw::setColorInt32( uint32_t clr ) {
    constexpr float i255 = 1/255.0f;
    uint8_t b = ( ( clr       ) & 0xFF );
    uint8_t g = ( ( clr >> 8  ) & 0xFF );
    uint8_t r = ( ( clr >> 16 ) & 0xFF );
    uint8_t a = (   clr >> 24          );
    glColor4f( i255*r, i255*g, i255*b, i255*a );
    //printf( " r %i g %i b %i a %i     %f %f %f %f  \n", r, g, b, a,  i255*r, i255*g, i255*b, i255*a   );
};
*/

void billboardCam( ){
    float glMat[16];
    //glMatrixMode(GL_MODELVIEW);
    glGetFloatv(GL_MODELVIEW_MATRIX , glMat);
    glMat[0 ] = 1;   glMat[1 ] = 0;   glMat[2 ] = 0;
    glMat[4 ] = 0;   glMat[5 ] = 1;   glMat[6 ] = 0;
    glMat[8 ] = 0;   glMat[9 ] = 0;   glMat[10] = 1;
    glLoadMatrixf(glMat);
}

void billboardCamProj( ){
    float glCam  [16];
    float glModel[16];
    glGetFloatv (GL_MODELVIEW_MATRIX,  glModel);
    glGetFloatv (GL_PROJECTION_MATRIX, glCam);
    //glMatrixMode(GL_MODELVIEW);

    Mat3f mat;
    mat.a.set(glCam[0],glCam[1],glCam[2]);       mat.a.mul(1/mat.a.norm2());
    mat.b.set(glCam[4],glCam[5],glCam[6]);       mat.b.mul(1/mat.b.norm2());
    mat.c.set(glCam[8],glCam[9],glCam[10]);      mat.c.mul(1/mat.c.norm2());

    glModel[0 ] = mat.a.x;   glModel[1 ] = mat.b.x;   glModel[2 ] = mat.c.x;
    glModel[4 ] = mat.a.y;   glModel[5 ] = mat.b.y;   glModel[6 ] = mat.c.y;
    glModel[8 ] = mat.a.z;   glModel[9 ] = mat.b.z;   glModel[10] = mat.c.z;

    //glModel[0 ] = glCam[0];   glModel[1 ] = glCam[4];   glModel[2 ] = glCam[8];
    //glModel[4 ] = glCam[1];   glModel[5 ] = glCam[5];   glModel[6 ] = glCam[9];
    //glModel[8 ] = glCam[2];   glModel[9 ] = glCam[6];   glModel[10] = glCam[10];

    glLoadMatrixf(glModel);
}

void text( const char * str, int itex, float sz, int iend ){
    const int nchars = 95;
    float persprite = 1.0f/nchars;
    glEnable     ( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, itex );
    glEnable(GL_BLEND);
    glEnable(GL_ALPHA_TEST);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    int terminator = 0xFFFF;
    if(iend<=0) { terminator=-iend; iend=256; };
    for(int i=0; i<iend; i++){
        if  (str[i]==terminator) break;
        int isprite = str[i] - 33;
        float offset  = isprite*persprite+(persprite*0.57);
        float xi = i*sz;
        glTexCoord2f( offset          , 1.0f ); glVertex3f( xi   ,    0, 0.0f );
        glTexCoord2f( offset+persprite, 1.0f ); glVertex3f( xi+sz,    0, 0.0f );
        glTexCoord2f( offset+persprite, 0.0f ); glVertex3f( xi+sz, sz*2, 0.0f );
        glTexCoord2f( offset          , 0.0f ); glVertex3f( xi   , sz*2, 0.0f );
    }
    glEnd();
    glDisable  ( GL_BLEND );
    glDisable  ( GL_ALPHA_TEST );
    glDisable  ( GL_TEXTURE_2D );
    //glBlendFunc( GL_ONE, GL_ZERO );
}

void text( const char * str, int itex, float sz, Vec2i block_size ){
    const int nchars = 95;
    float persprite = 1.0f/nchars;
    glEnable     ( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, itex );
    glEnable(GL_BLEND);
    glEnable(GL_ALPHA_TEST);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    //int terminator = 0xFFFF;
    //if(iend<=0) { terminator=-iend; iend=256; };
    char terminator = '\0';
    int iline=0,ix=0;
    //printf("\n"); printf("-------\n");
    for(int i=0; i<65536; i++){
        char ch = str[i]; // printf("%c", ch);
        if       (ch==terminator){ break; }
        else if ((ch=='\n')||(ix>block_size.x)){ iline++; ix=0; if(iline>block_size.y) break; continue; }
        int isprite = ch - 33;
        float offset  = isprite*persprite+(persprite*0.57);
        float x = ix   *sz;
        float y = -iline*sz*2;
        glTexCoord2f( offset          , 1.0f ); glVertex3f( x   , y+   0, 0.0f );
        glTexCoord2f( offset+persprite, 1.0f ); glVertex3f( x+sz, y+   0, 0.0f );
        glTexCoord2f( offset+persprite, 0.0f ); glVertex3f( x+sz, y+sz*2, 0.0f );
        glTexCoord2f( offset          , 0.0f ); glVertex3f( x   , y+sz*2, 0.0f );
        ix++;
    }
    glEnd();
    glDisable  ( GL_BLEND );
    glDisable  ( GL_ALPHA_TEST );
    glDisable  ( GL_TEXTURE_2D );
    glBlendFunc( GL_ONE, GL_ZERO );
}








//



void point( const Vec3f& vec ){
    //glDisable (GL_LIGHTING);
    glBegin   (GL_POINTS);
        glVertex3d( vec.x, vec.y, vec.z );
    glEnd();
}

void pointCross( const Vec3f& vec, float sz ){
    //glDisable (GL_LIGHTING);
    glBegin   (GL_LINES);
        glVertex3f( vec.x-sz, vec.y, vec.z ); glVertex3f( vec.x+sz, vec.y, vec.z );
        glVertex3f( vec.x, vec.y-sz, vec.z ); glVertex3f( vec.x, vec.y+sz, vec.z );
        glVertex3f( vec.x, vec.y, vec.z-sz ); glVertex3f( vec.x, vec.y, vec.z+sz );
    glEnd();
}

void pointCross( const Vec3f& vec, double sz ){
    //glDisable (GL_LIGHTING);
    glBegin   (GL_LINES);
        glVertex3d( vec.x-sz, vec.y, vec.z ); glVertex3d( vec.x+sz, vec.y, vec.z );
        glVertex3d( vec.x, vec.y-sz, vec.z ); glVertex3d( vec.x, vec.y+sz, vec.z );
        glVertex3d( vec.x, vec.y, vec.z-sz ); glVertex3d( vec.x, vec.y, vec.z+sz );
    glEnd();
}

void vec( const Vec3f& vec ){
    //glDisable (GL_LIGHTING);
    glBegin   (GL_LINES);
        glVertex3d( 0, 0, 0 ); glVertex3d( vec.x, vec.y, vec.z );
    glEnd();
}

void vecInPos( const Vec3f& v, const Vec3f& pos ){
    //glDisable (GL_LIGHTING);
    glBegin   (GL_LINES);
        glVertex3d( pos.x, pos.y, pos.z ); glVertex3d( pos.x+v.x, pos.y+v.y, pos.z+v.z );
    glEnd();
}

void line( const Vec3f& p1, const Vec3f& p2 ){
    //glDisable (GL_LIGHTING);
    glBegin   (GL_LINES);
        glVertex3d( p1.x, p1.y, p1.z ); glVertex3d( p2.x, p2.y, p2.z );
    glEnd();
}

void arrow( const Vec3f& p1, const Vec3f& p2, float sz ){
    //glDisable (GL_LIGHTING);
    Vec3f up,lf,p;
    Vec3f fw = p2-p1; fw.normalize();
    fw.getSomeOrtho(up,lf);
    fw.mul(sz); lf.mul(sz); up.mul(sz);
    glBegin   (GL_LINES);
        glVertex3d( p1.x, p1.y, p1.z ); glVertex3d( p2.x, p2.y, p2.z );
        p = p2 - fw + up; glVertex3d( p.x, p.y, p.z ); glVertex3d( p2.x, p2.y, p2.z );
        p = p2 - fw - up; glVertex3d( p.x, p.y, p.z ); glVertex3d( p2.x, p2.y, p2.z );
        p = p2 - fw + lf; glVertex3d( p.x, p.y, p.z ); glVertex3d( p2.x, p2.y, p2.z );
        p = p2 - fw - lf; glVertex3d( p.x, p.y, p.z ); glVertex3d( p2.x, p2.y, p2.z );
    glEnd();
}

void polyLine( int n, Vec3d * ps, bool closed ){   // closed=false
    //printf("%i %i\n", n, closed );
    if(closed){ glBegin(GL_LINE_LOOP); }else{ glBegin(GL_LINE_STRIP); }
    for(int i=0; i<n; i++){
        //printf("%i (%3.3f,%3.3f,%3.3f)\n", i, ps[i].x, ps[i].y, ps[i].z );
        glVertex3d( ps[i].x, ps[i].y, ps[i].z );
    };
    glEnd();
}

void scale( const Vec3f& p1, const Vec3f& p2, const Vec3f& up, float tick, float sza, float szb ){
    //glDisable (GL_LIGHTING);
    Vec3f d,a,b,p;
    d.set_sub( p2, p1 );
    float L = d.norm();
    int n = (L+0.0001)/tick;
    d.mul( 1/L );
    a.set(up);
    a.add_mul( d, -d.dot( a ) );
    b.set_cross( d, a  );
    glBegin   (GL_LINES);
    p.set(p1);
    d.mul( L/n );
    a.mul(sza);
    b.mul(szb);
    for( int i=0; i<=n; i++){
        glVertex3d( p.x-a.x, p.y-a.y, p.z-a.z );
        glVertex3d( p.x+a.x, p.y+a.y, p.z+a.z );
        glVertex3d( p.x-b.x, p.y-b.y, p.z-a.z );
        glVertex3d( p.x+b.x, p.y+b.y, p.z+b.z );
        p.add( d );
    }
    glVertex3d( p1.x, p1.y, p1.z ); glVertex3d( p2.x, p2.y, p2.z );
    glEnd();
}

void triangle( const Vec3f& p1, const Vec3f& p2, const Vec3f& p3 ){
    //printf("p1 (%3.3f,%3.3f,%3.3f) p2 (%3.3f,%3.3f,%3.3f) p3 (%3.3f,%3.3f,%3.3f) \n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z);
    Vec3f d1,d2,normal;
    d1.set( p2 - p1 );
    d2.set( p3 - p1 );
    normal.set_cross(d1,d2);
    normal.normalize();
    glBegin   (GL_TRIANGLES);
        glNormal3d( normal.x, normal.y, normal.z );
        glVertex3d( p1.x, p1.y, p1.z );
        glVertex3d( p2.x, p2.y, p2.z );
        glVertex3d( p3.x, p3.y, p3.z );
    glEnd();
    //drawPointCross( p1, 0.1 );
    //drawPointCross( p2, 0.1 );
    //drawPointCross( p3, 0.1 );
}

void matInPos( const Mat3f& mat, const Vec3f& pos ){
    //glDisable (GL_LIGHTING);
    glBegin   (GL_LINES);
        glColor3f( 1, 0, 0 ); glVertex3d( pos.x, pos.y, pos.z ); glVertex3d( pos.x+mat.xx, pos.y+mat.xy, pos.z+mat.xz );
        glColor3f( 0, 1, 0 ); glVertex3d( pos.x, pos.y, pos.z ); glVertex3d( pos.x+mat.yx, pos.y+mat.yy, pos.z+mat.yz );
        glColor3f( 0, 0, 1 ); glVertex3d( pos.x, pos.y, pos.z ); glVertex3d( pos.x+mat.zx, pos.y+mat.zy, pos.z+mat.zz );
    glEnd();
}

void shape( const Vec3f& pos, const Mat3f& rot, int ishape, bool trasposed ){
    glPushMatrix();
    float glMat[16];
    if( trasposed ){
        toGLMatT ( pos, rot, glMat );
    }else{
        toGLMat( pos, rot, glMat );
    }
    glMultMatrixf( glMat );
    glCallList( ishape );
    glPopMatrix();
}

void shape    ( const Vec3f& pos, const Quat4f& qrot, int ishape ){
    glPushMatrix();
    float glMat[16];
    toGLMat ( pos, qrot, glMat );
    glMultMatrixf( glMat );
    glCallList( ishape );
    glPopMatrix();
}

void shape    ( const Vec3f& pos, const Quat4f& qrot, const Vec3f& scale, int ishape ){
    glPushMatrix();
    float glMat[16];
    toGLMat ( pos, qrot, scale, glMat );
    glMultMatrixf( glMat );
    glCallList( ishape );
    glPopMatrix();
}

int coneFan( int n, float r, const Vec3f& base, const Vec3f& tip ){
    int nvert=0;
    Vec3f a,b,c,c_hat;
    c.set_sub( tip, base );
    c_hat.set_mul( c, 1/c.norm() );
    c_hat.getSomeOrtho( a, b );
    a.normalize();
    b.normalize();
    //float alfa = 2*M_PI/n;
    float alfa = 2*M_PI/n;
    Vec2f rot,drot;
    rot .set(1.0f,0.0f);
    drot.set( cos( alfa ), sin( alfa ) );

    Vec3f q; q.set(c); q.add_mul( a, -r );
    float pnab =  c_hat.dot( q )/q.norm();
    float pnc  =  sqrt( 1 - pnab*pnab );

    glBegin   ( GL_TRIANGLE_FAN );
    //glBegin   ( GL_LINES );
    //glBegin   ( GL_LINE_STRIP );

    glNormal3f( c_hat.x, c_hat.z, c_hat.z );
    //printf( "pn0 %f %f %f \n", c_hat.x, c_hat.z, c_hat.z );
    glVertex3f( tip.x, tip.y, tip.z ); nvert++;
    for(int i=0; i<=n; i++ ){
        Vec3f p,pn;
        p .set( rot.x*a.x +  rot.y*b.x, rot.x*a.y + rot.y*b.y, rot.x*a.z + rot.y*b.z    );
        pn.set( pnab*p.x + pnc*c_hat.x, pnab*p.y + pnc*c_hat.y, pnab*p.z + pnc*c_hat.z  );

        glNormal3f( pn.x, pn.y, pn.z );
        glVertex3f( base.x + r*p.x, base.y + r*p.y, base.z + r*p.z ); nvert++;
        rot.mul_cmplx( drot );
    }
    glEnd();
    return nvert;
}

int cylinderStrip( int n, float r1, float r2, const Vec3f& base, const Vec3f& tip ){
    int nvert=0;

    Vec3f a,b,c,c_hat;
    c.set_sub( tip, base );
    c_hat.set_mul( c, 1/c.norm() );
    c_hat.getSomeOrtho( a, b );
    a.normalize();
    b.normalize();

    float alfa = 2*M_PI/n;
    Vec2f rot,drot;
    rot .set(1.0f,0.0f);
    drot.set( cos( alfa ), sin( alfa ) );

    Vec3f q; q.set(c); q.add_mul( a, -(r1-r2) );
    float pnab =  c_hat.dot( q )/q.norm();
    float pnc  =  sqrt( 1 - pnab*pnab );

    glBegin   ( GL_TRIANGLE_STRIP );
    //glBegin   ( GL_LINES );
    for(int i=0; i<=n; i++ ){
        Vec3f p,pn;
        p .set( rot.x*a.x +  rot.y*b.x, rot.x*a.y + rot.y*b.y, rot.x*a.z + rot.y*b.z    );
        pn.set( pnab*p.x + pnc*c_hat.x, pnab*p.y + pnc*c_hat.y, pnab*p.z + pnc*c_hat.z  );
        //printf( "p %f %f %f   pn %f %f %f |pn| %f \n", p.x, p.y, p.z,   pn.x, pn.y, pn.z, pn.norm() );
        glNormal3f( pn.x, pn.y, pn.z );
        glVertex3f( base.x + r1*p.x, base.y + r1*p.y, base.z + r1*p.z ); nvert++;
        glVertex3f( tip .x + r2*p.x, tip .y + r2*p.y, tip .z + r2*p.z ); nvert++;
        rot.mul_cmplx( drot );
    }
    glEnd();
    return nvert;
}

int cylinderStrip_wire( int n, float r1, float r2, const Vec3f& base, const Vec3f& tip ){
    int nvert=0;

    Vec3f a,b,c,c_hat;
    c.set_sub( tip, base );
    c_hat.set_mul( c, 1/c.norm() );
    c_hat.getSomeOrtho( a, b );
    a.normalize();
    b.normalize();

    float alfa = 2*M_PI/n;
    Vec2f rot,drot;
    rot .set(1.0f,0.0f);
    drot.set( cos( alfa ), sin( alfa ) );

    Vec3f q; q.set(c); q.add_mul( a, -(r1-r2) );
    float pnab =  c_hat.dot( q )/q.norm();
    float pnc  =  sqrt( 1 - pnab*pnab );

    glBegin   ( GL_LINE_LOOP );
    //glBegin   ( GL_LINES );
    for(int i=0; i<n; i++ ){
        Vec3f p;
        p .set( rot.x*a.x +  rot.y*b.x, rot.x*a.y + rot.y*b.y, rot.x*a.z + rot.y*b.z    );
        glVertex3f( base.x + r1*p.x, base.y + r1*p.y, base.z + r1*p.z ); nvert++;
        glVertex3f( tip .x + r2*p.x, tip .y + r2*p.y, tip .z + r2*p.z ); nvert++;
        rot.mul_cmplx( drot );
    }
    for(int i=0; i<n; i++ ){
        Vec3f p;
        p .set( rot.x*a.x +  rot.y*b.x, rot.x*a.y + rot.y*b.y, rot.x*a.z + rot.y*b.z    );
        glVertex3f( base.x + r1*p.x, base.y + r1*p.y, base.z + r1*p.z ); nvert++;
        rot.mul_cmplx( drot );
    }
    for(int i=0; i<n; i++ ){
        Vec3f p;
        p .set( rot.x*a.x +  rot.y*b.x, rot.x*a.y + rot.y*b.y, rot.x*a.z + rot.y*b.z    );
        glVertex3f( tip .x + r2*p.x, tip .y + r2*p.y, tip .z + r2*p.z ); nvert++;
        rot.mul_cmplx( drot );
    }
    glEnd();
    return nvert;
}

int cone( int n, float phi0, float phi1, float r1, float r2, const Vec3f& base, const Vec3f& tip, bool smooth ){
    int nvert=0;

    Vec3f a,b,c,c_hat;
    c.set_sub( tip, base );
    c_hat.set_mul( c, 1/c.norm() );
    c_hat.getSomeOrtho( a, b );
    a.normalize();
    b.normalize();

    //float alfa = 2*M_PI/n;
    float alfa = (phi1-phi0)/n;
    Vec2f rot,drot;
    //rot .set(1.0f,0.0f);
    rot.set( cos( phi0 ), sin( phi0 ) );
    drot.set( cos( alfa ), sin( alfa ) );

    Vec3f q; q.set(c); q.add_mul( a, -(r1-r2) );
    float pnab =  c_hat.dot( q )/q.norm();
    float pnc  =  sqrt( 1 - pnab*pnab );

    glBegin   ( GL_QUADS );
    //glBegin   ( GL_LINES );
    Vec3f p,pn,op,opn;
    op .set( rot.x*a.x +  rot.y*b.x, rot.x*a.y + rot.y*b.y, rot.x*a.z + rot.y*b.z    );
    opn.set( pnab*op.x + pnc*c_hat.x, pnab*op.y + pnc*c_hat.y, pnab*op.z + pnc*c_hat.z  );
    if( smooth ){
        for(int i=0; i<n; i++ ){

            glNormal3f( opn.x, opn.y, opn.z );		glVertex3f( base.x + r1*op.x, base.y + r1*op.y, base.z + r1*op.z ); nvert++;
            glNormal3f( opn.x, opn.y, opn.z );		glVertex3f( tip .x + r2*op.x, tip .y + r2*op.y, tip .z + r2*op.z ); nvert++;

            rot.mul_cmplx( drot );
            p .set( rot.x*a.x +  rot.y*b.x, rot.x*a.y + rot.y*b.y, rot.x*a.z + rot.y*b.z    );
            pn.set( pnab*p.x + pnc*c_hat.x, pnab*p.y + pnc*c_hat.y, pnab*p.z + pnc*c_hat.z  );
            pn.normalize();

            glNormal3f( pn.x, pn.y, pn.z );
            glVertex3f( tip .x + r2*p.x, tip .y + r2*p.y, tip .z + r2*p.z ); nvert++;
            glVertex3f( base.x + r1*p.x, base.y + r1*p.y, base.z + r1*p.z ); nvert++;

            op.set(p);
            opn.set(pn);

        }
    }else{
        for(int i=0; i<n; i++ ){

            //printf( " %i (%3.3f,%3.3f) \n", i, rot.x, rot.y );

            rot.mul_cmplx( drot );

            p .set( rot.x*a.x +  rot.y*b.x, rot.x*a.y + rot.y*b.y, rot.x*a.z + rot.y*b.z    );
            pn.set( pnab*p.x + pnc*c_hat.x, pnab*p.y + pnc*c_hat.y, pnab*p.z + pnc*c_hat.z  );

            Vec3f normal; normal.set_add( opn, pn ); normal.normalize();

            glNormal3f( normal.x, normal.y, normal.z );
            glVertex3f( base.x + r1*op.x, base.y + r1*op.y, base.z + r1*op.z ); nvert++;
            glVertex3f( tip .x + r2*op.x, tip .y + r2*op.y, tip .z + r2*op.z ); nvert++;

            glVertex3f( tip .x + r2*p.x, tip .y + r2*p.y, tip .z + r2*p.z ); nvert++;
            glVertex3f( base.x + r1*p.x, base.y + r1*p.y, base.z + r1*p.z ); nvert++;

            op.set(p);
            opn.set(pn);

        }
    }
    glEnd();
    return nvert;
}

int sphereTriangle( int n, float r, const Vec3f& pos, const Vec3f& a, const Vec3f& b, const Vec3f& c ){
    int nvert=0;
    float d = 1.0f/n;
    Vec3f da,db;
    da.set_sub( a, c ); da.mul( d );
    db.set_sub( b, c ); db.mul( d );
    for( int ia=0; ia<n; ia++ ){
        Vec3f p0,p; p0.set( c );
        p0.add_mul( da, ia );
        p.set_mul( p0, 1.0f/p0.norm() );
        glBegin   (GL_TRIANGLE_STRIP);
        //glBegin   (GL_LINES);
        //glColor3f( d*ia, 0, 0 );
        glNormal3f( p.x, p.y, p.z );
        glVertex3f( r*p.x+pos.x, r*p.y+pos.y, r*p.z+pos.z );   nvert++;
        //glVertex3f( r*p.x+pos.x+p.x, r*p.y+pos.y+p.y, r*p.z+pos.z+p.z );
        for( int ib=0; ib<(n-ia); ib++ ){
            Vec3f p;
            p.set_add( p0, da );
            p.normalize();
            //glColor3f( 0, 1, 0 );
            glNormal3f( p.x, p.y, p.z );
            glVertex3f( r*p.x+pos.x, r*p.y+pos.y, r*p.z+pos.z );   nvert++;
            //glVertex3f( r*p.x+pos.x+p.x, r*p.y+pos.y+p.y, r*p.z+pos.z+p.z );
            p.set_add( p0, db );
            p.normalize();
            //glColor3f( 0, 0, 1 );
            glNormal3f( p.x, p.y, p.z );
            glVertex3f( r*p.x+pos.x, r*p.y+pos.y, r*p.z+pos.z );   nvert++;
            //glVertex3f( r*p.x+pos.x+p.x, r*p.y+pos.y+p.y, r*p.z+pos.z+p.z );
            p0.add( db );
            //printf(" %f %f %f %f \n", p.x, p.y, p.z, p.norm() );
        }
        glEnd();
    }
    return nvert;
}

int sphereTriangle_wire( int n, float r, const Vec3f& pos, const Vec3f& a, const Vec3f& b, const Vec3f& c ){
    int nvert=0;
    float d = 1.0f/n;
    Vec3f da,db;
    da.set_sub( a, c ); da.mul( d );
    db.set_sub( b, c ); db.mul( d );
    for( int ia=0; ia<n; ia++ ){
        Vec3f p0,p; p0.set( c );
        p0.add_mul( da, ia );
        glBegin   (GL_LINE_STRIP); //glColor3f(0.0,0.0,1.0);
        p.set(p0); p.normalize();
        glVertex3f( r*p.x+pos.x, r*p.y+pos.y, r*p.z+pos.z );   nvert++;
        for( int ib=0; ib<(n-ia); ib++ ){
            p.set_add( p0, da ); p.normalize();
            glVertex3f( r*p.x+pos.x, r*p.y+pos.y, r*p.z+pos.z );   nvert++;
            p.set_add( p0, db ); p.normalize();
            glVertex3f( r*p.x+pos.x, r*p.y+pos.y, r*p.z+pos.z );   nvert++;
            p0.add( db );
        }
        glEnd();
        glBegin   (GL_LINE_STRIP);         //glColor3f(1.0,0.0,0.0);
        for( int ib=0; ib<=(n-ia); ib++ ){
            //p.set_add( p0, da );
            p.set(p0); p.normalize();
            glVertex3f( r*p.x+pos.x, r*p.y+pos.y, r*p.z+pos.z );   nvert++;
            p0.sub( db );
        }
        glEnd();
    }
    return nvert;
}

int sphere_oct( int n, float r, const Vec3f& pos, bool wire ){
    int nvert=0;
    Vec3f px,mx,py,my,pz,mz;
    px.set( 1,0,0); py.set(0, 1,0); pz.set(0,0, 1);
    mx.set(-1,0,0); my.set(0,-1,0); mz.set(0,0,-1);
    if(wire){
        nvert += sphereTriangle_wire( n, r, pos, mz, mx, my );
        nvert += sphereTriangle_wire( n, r, pos, mz, my, px );
        nvert += sphereTriangle_wire( n, r, pos, mz, px, py );
        nvert += sphereTriangle_wire( n, r, pos, mz, py, mx );
        nvert += sphereTriangle_wire( n, r, pos, pz, mx, my );
        nvert += sphereTriangle_wire( n, r, pos, pz, my, px );
        nvert += sphereTriangle_wire( n, r, pos, pz, px, py );
        nvert += sphereTriangle_wire( n, r, pos, pz, py, mx );
    }else{
        nvert += sphereTriangle( n, r, pos, mz, mx, my );
        nvert += sphereTriangle( n, r, pos, mz, my, px );
        nvert += sphereTriangle( n, r, pos, mz, px, py );
        nvert += sphereTriangle( n, r, pos, mz, py, mx );
        nvert += sphereTriangle( n, r, pos, pz, mx, my );
        nvert += sphereTriangle( n, r, pos, pz, my, px );
        nvert += sphereTriangle( n, r, pos, pz, px, py );
        nvert += sphereTriangle( n, r, pos, pz, py, mx );
    }
    return nvert;
}

int capsula( Vec3f p0, Vec3f p1, float r1, float r2, float theta1, float theta2, float dTheta, int nPhi, bool capped ){
    Vec3f ax   = p1-p0;  float L = ax.normalize();
    Vec3f up,left;       ax.getSomeOrtho(up,left);
    Vec2f cph=Vec2fX, dph;
    dph.fromAngle( 2*M_PI/nPhi );
    // Cylinder
    Vec2f cth,dth;
    float dr = (r2-r1);
    float cv = sqrt(L*L+dr*dr);
    cth.set( L/cv, -dr/cv );
    int nvert=0;
    glBegin(GL_TRIANGLE_STRIP);
    for(int iph=0; iph<(nPhi+1); iph++){
        Vec3f pa = p0 + left*(cph.x*r1) + up*(cph.y*r1);
        Vec3f pb = p1 + left*(cph.x*r2) + up*(cph.y*r2);
        Vec3f na = (left*(cph.x) + up*(cph.y)*cth.x + ax*cth.y)*-1.0;
        Vec3f nb = (left*(cph.x) + up*(cph.y)*cth.x + ax*cth.y)*-1.0;
        glNormal3f(na.x,na.y,na.z); glVertex3f(pa.x,pa.y,pa.z); nvert++;
        glNormal3f(nb.x,nb.y,nb.z); glVertex3f(pb.x,pb.y,pb.z); nvert++;
        cph.mul_cmplx(dph);
    }
    glEnd();

    float DTh,h;
    int nTheta;
    // Spherical Cap
    cph=Vec2fX;
    cth.set( L/cv, -dr/cv );
    //dth.fromSin(v1/r1);
    DTh = (-theta1 - asin(cth.y));
    nTheta = (int)(fabs(DTh)/dTheta);
    dth.fromAngle( DTh/nTheta );
    //printf( " cth (%f,%f)  dth (%f,%f) \n", cth.x, cth.y,  dth.x, dth.y );
    r1/=cth.x;
    h  =-cth.y*r1;
    // Left
    for(int ith=0; ith<(nTheta+1); ith++){
        Vec2f cth_ = Vec2f::mul_cmplx(cth,dth);
        glBegin(GL_TRIANGLE_STRIP);
        //glBegin(GL_LINES);
        for(int iph=0; iph<(nPhi+1); iph++){
            Vec3f pa = p0 + (left*(cph.x*r1) + up*(cph.y*r1))*cth.x  + ax*(h+cth.y*r1);
            Vec3f pb = p0 + (left*(cph.x*r1) + up*(cph.y*r1))*cth_.x + ax*(h+cth_.y*r1);
            Vec3f na = (left*(cph.x) + up*(cph.y)*cth.x  + ax*cth.y)*1.0;
            Vec3f nb = (left*(cph.x) + up*(cph.y)*cth_.x + ax*cth_.y)*1.0;
            glNormal3f(na.x,na.y,na.z); glVertex3f(pa.x,pa.y,pa.z); nvert++;
            glNormal3f(nb.x,nb.y,nb.z); glVertex3f(pb.x,pb.y,pb.z); nvert++;
            //na.mul(0.2);
            //glVertex3f(pa.x,pa.y,pa.z);   glVertex3f(pa.x+na.x,pa.y+na.y,pa.z+na.z);
            cph.mul_cmplx(dph);
        }
        glEnd();
        //printf( "%i cth (%f,%f)  cth_ (%f,%f) \n", ith, cth.x, cth.y,  cth_.x, cth_.y );
        cth=cth_;
    }
    //return 0;
    cph=Vec2fX;
    cth.set( L/cv, -dr/cv );
    //cth = Vec2fX;
    //cth.set( dr/cv, L/cv);
    //dth.fromAngle( asin(v2/r2)/nTheta );
    DTh    = (theta2-asin(cth.y));
    nTheta = (int)(fabs(DTh)/dTheta);
    dth.fromAngle(DTh/nTheta );
    r2/= cth.x;
    h  =-cth.y*r2;
    // Right
    for(int ith=0; ith<(nTheta+1); ith++){
        Vec2f cth_ = Vec2f::mul_cmplx(cth,dth);
        glBegin(GL_TRIANGLE_STRIP);
        for(int iph=0; iph<(nPhi+1); iph++){
            Vec3f pa = p1 + (left*(cph.x*r2) + up*(cph.y*r2))*cth.x  + ax*(h+cth.y*r2);
            Vec3f pb = p1 + (left*(cph.x*r2) + up*(cph.y*r2))*cth_.x + ax*(h+cth_.y*r2);
            Vec3f na = (left*(cph.x) + up*(cph.y)*cth.x  + ax*cth.y)*-1.0;
            Vec3f nb = (left*(cph.x) + up*(cph.y)*cth_.x + ax*cth_.y)*-1.0;
            glNormal3f(na.x,na.y,na.z); glVertex3f(pa.x,pa.y,pa.z); nvert++;
            glNormal3f(nb.x,nb.y,nb.z); glVertex3f(pb.x,pb.y,pb.z); nvert++;
            cph.mul_cmplx(dph);
        }
        glEnd();
        cth=cth_;
    }
    return nvert;
}

int paraboloid     ( Vec3f p0, Vec3f ax, float r, float l, float nR, int nPhi, bool capped ){
    float L = ax.normalize();
    Vec3f up,left;       ax.getSomeOrtho(up,left);
    Vec2f cph=Vec2fX, dph;
    dph.fromAngle( 2*M_PI/nPhi );
    float dr = r/nR;
    float a  = 1.0; // TODO
    int nvert=0;
    for(int ir=0; ir<nR; ir++){
        glBegin(GL_TRIANGLE_STRIP);
        //glBegin(GL_LINES);
        float r1 =(ir-1)*dr;
        float r2 =(ir  )*dr;
        for(int iph=0; iph<(nPhi+1); iph++){
            float h1 = a*r1*r1;
            float h2 = a*r2*r2;
            Vec3f pa = p0 + left*(cph.x*r1) + up*(cph.y*r1) + ax*h1;
            Vec3f pb = p0 + left*(cph.x*r2) + up*(cph.y*r2) + ax*h2;
            //Vec3f na = left*(cph.x) + up*(cph.y)*cth.x  + ax*cth.;
            //Vec3f nb = left*(cph.x) + up*(cph.y)*cth_.x + ax*cth_.y;
            //glNormal3f(na.x,na.y,na.z);
            glVertex3f(pa.x,pa.y,pa.z); nvert++;
            //glNormal3f(nb.x,nb.y,nb.z);
            glVertex3f(pb.x,pb.y,pb.z); nvert++;
            cph.mul_cmplx(dph);
        }
        glEnd();
    }
    return nvert;
}

int circleAxis( int n, const Vec3f& pos, const Vec3f& v0, const Vec3f& uaxis, float R, float dca, float dsa ){
    Vec3f v; v.set(v0);
    glBegin( GL_LINE_LOOP );
    for( int i=0; i<n; i++ ){
        glVertex3f( pos.x+v.x*R, pos.y+v.y*R, pos.z+v.z*R );
        //printf( " drawCircleAxis %i (%3.3f,%3.3f,%3.3f) \n", i, v.x, v.y, v.z );
        v.rotate_csa( dca, dsa, uaxis );
    }
    glEnd();
    return n;
}

int circleAxis( int n, const Vec3f& pos, const Vec3f& v0, const Vec3f& uaxis, float R ){
    float dphi = 2*M_PI/n;
    float dca  = cos( dphi );
    float dsa  = sin( dphi );
    return circleAxis( n, pos, v0, uaxis, R, dca, dsa );
}

int sphereOctLines( int n, float R, const Vec3f& pos ){
	int nvert=0;
    float dphi = 2*M_PI/n;
    float dca  = cos( dphi );
    float dsa  = sin( dphi );
    nvert += circleAxis( n, pos, {0,1,0}, {1.0d,0.0d,0.0d}, R, dca, dsa );
    nvert += circleAxis( n, pos, {0,0,1}, {0.0d,1.0d,0.0d}, R, dca, dsa );
    nvert += circleAxis( n, pos, {1,0,0}, {0.0d,0.0d,1.0d}, R, dca, dsa );
	return nvert;
}

void planarPolygon( int n, const int * inds, const Vec3d * points ){
    if( n < 3 ) return;

    Vec3f a,b,c,normal;
    a = (Vec3f) points[inds[0]];
    b = (Vec3f) points[inds[1]];
    c = (Vec3f) points[inds[2]];
    normal.set_cross( a-b, b-c );
    normal.normalize( );

    glBegin( GL_TRIANGLE_FAN );
    glNormal3f( normal.x, normal.y, normal.z );
    glVertex3f( a.x, a.y, a.z );
    glVertex3f( b.x, b.y, b.z );
    glVertex3f( c.x, c.y, c.z );
    for( int i=3; i<n; i++ ){
        convert( points[inds[i]], a );
        glVertex3f( a.x, a.y, a.z );
        //average.add( a );
    }
    glEnd();
}

void polygonNormal( int n, const int * inds, const Vec3d * points ){
    if( n < 3 ) return;

    Vec3f a,b,c,normal;
    a = (Vec3f) points[inds[0]];
    b = (Vec3f) points[inds[1]];
    c = (Vec3f) points[inds[2]];
    normal.set_cross( a-b, b-c );
    normal.normalize( );

    glBegin( GL_LINES );
        glVertex3f( a.x, a.y, a.z );
        glVertex3f( a.x+normal.x, a.y+normal.y, a.z+normal.z );
    glEnd();
}

void polygonBorder( int n, const int * inds, const Vec3d * points ){
    glBegin( GL_LINE_LOOP );
    Vec3f a;
    for( int i=0; i<n; i++ ){
        convert( points[inds[i]], a );
        glVertex3f( a.x, a.y, a.z );
    }
    glEnd();
}


// void planarPolygon( int ipl, Mesh& mesh ){
//     Polygon * pl = mesh.polygons[ipl];
//     Draw3D:: drawPlanarPolygon( pl->ipoints.size(), &pl->ipoints.front(), &mesh.points.front() );
// }

// void polygonNormal( int ipl, Mesh& mesh ){
//     Polygon * pl = mesh.polygons[ipl];
//     Draw3D:: drawPolygonNormal( pl->ipoints.size(), &pl->ipoints.front(), &mesh.points.front() );
// }

// void polygonBorder( int ipl, Mesh& mesh ){
//     Polygon * pl = mesh.polygons[ipl];
//     Draw3D:: drawPolygonBorder( pl->ipoints.size(), &pl->ipoints.front(), &mesh.points.front() );
// }


void points( int n, const  Vec3d * points, float sz ){
    if(sz<=0){
        glBegin( GL_POINTS );
        for( int i=0; i<n; i++ ){
            Vec3f a;
            convert( points[i], a );
            glVertex3f( a.x, a.y, a.z );
        }
        glEnd();
	}else{
        glBegin( GL_LINES );
        for( int i=0; i<n; i++ ){
            Vec3f vec;
            convert( points[i], vec );
            glVertex3f( vec.x-sz, vec.y, vec.z ); glVertex3f( vec.x+sz, vec.y, vec.z );
            glVertex3f( vec.x, vec.y-sz, vec.z ); glVertex3f( vec.x, vec.y+sz, vec.z );
            glVertex3f( vec.x, vec.y, vec.z-sz ); glVertex3f( vec.x, vec.y, vec.z+sz );
        }
        glEnd();
	}
}

void lines( int nlinks, const  int * links, const  Vec3d * points ){
    int n2 = nlinks<<1;
    glBegin( GL_LINES );
    for( int i=0; i<n2; i+=2 ){
        //drawLine( points[links[i]], points[links[i+1]] );
        //printf ( " %i %i %i %f %f \n", i, links[i], links[i+1], points[links[i]].x, points[links[i+1]].x );
        Vec3f a,b;
        convert( points[links[i  ]], a );
        convert( points[links[i+1]], b );
        glVertex3f( a.x, a.y, a.z );
        glVertex3f( b.x, b.y, b.z );
    }
    glEnd();
}

    void triangles( int nlinks, const int * links, const Vec3d * points ){
        int n2 = nlinks*3;
        glBegin( GL_TRIANGLES );
        for( int i=0; i<n2; i+=3 ){
            //drawTriangle( points[links[i]], points[links[i+1]], points[links[i+2]] );
            //printf ( " %i %i %i %f %f \n", i, links[i], links[i+1], points[links[i]].x, points[links[i+1]].x );
            Vec3f a,b,c,normal;
            convert( points[links[i  ]], a );
            convert( points[links[i+1]], b );
            convert( points[links[i+2]], c );
            //printf( " %i (%3.3f,%3.3f,%3.3f) (%3.3f,%3.3f,%3.3f) (%3.3f,%3.3f,%3.3f) \n", i, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z  );
            normal.set_cross( a-b, b-c );
            normal.normalize( );
            glNormal3f( normal.x, normal.y, normal.z );
            glVertex3f( a.x, a.y, a.z );
            glVertex3f( b.x, b.y, b.z );
            glVertex3f( c.x, c.y, c.z );
        }
        glEnd();
    }


    void polygons( int nlinks, const int * ns, const int * links, const Vec3d * points ){
        const int * inds = links;
        for( int i=0; i<nlinks; i++ ){
            int ni = ns[i];
            planarPolygon( ni, inds, points );
            inds += ni;
        }
    }

    void kite( const Vec3f& pos, const Mat3f& rot, double sz ){
        glBegin  (GL_QUADS);
            glNormal3d( rot.b.x, rot.b.y, rot.b.z );
            glVertex3d( pos.x-sz*rot.a.x, pos.y-sz*rot.a.y, pos.z-sz*rot.a.z );
            glVertex3d( pos.x-sz*rot.c.x, pos.y-sz*rot.c.y, pos.z-sz*rot.c.z );
            glVertex3d( pos.x+sz*rot.a.x, pos.y+sz*rot.a.y, pos.z+sz*rot.a.z );
            glVertex3d( pos.x+sz*rot.c.x, pos.y+sz*rot.c.y, pos.z+sz*rot.c.z );
        glEnd();
    }

    void panel( const Vec3f& pos, const Mat3f& rot, const Vec2f& sz ){
        glBegin  (GL_QUADS);
            glNormal3f( rot.b.x, rot.b.y, rot.b.z );
            Vec3f p;
            glTexCoord2f(0.0,1.0); p=pos-rot.a*sz.a + rot.c*sz.b; glVertex3f( p.x, p.y, p.z );
            glTexCoord2f(0.0,0.0); p=pos-rot.a*sz.a - rot.c*sz.b; glVertex3f( p.x, p.y, p.z );
            glTexCoord2f(1.0,0.0); p=pos+rot.a*sz.a - rot.c*sz.b; glVertex3f( p.x, p.y, p.z );
            glTexCoord2f(1.0,1.0); p=pos+rot.a*sz.a + rot.c*sz.b; glVertex3f( p.x, p.y, p.z );
        glEnd();
    }

    void vectorArray(int n, Vec3d* ps, Vec3d* vs, double sc ){
        glBegin(GL_LINES);
        for(int i=0; i<n; i++){
            Vec3d p=ps[i];        glVertex3f(p.x,p.y,p.z);
            p.add_mul( vs[i], sc); glVertex3f(p.x,p.y,p.z);
        }
        glEnd();
    }

    void scalarArray(int n, Vec3d* ps, double* vs, double vmin, double vmax ){
        glBegin(GL_POINTS);
        double sc = 1/(vmax-vmin);
        for(int i=0; i<n; i++){
            Vec3d p=ps[i];
            double c = (vs[i]-vmin)*sc;
            glColor3f(c,c,c);
            glVertex3f(p.x,p.y,p.z);
            //printf( "i %i p(%g,%g,%g) v: %g c: %g\n", i, p.x,p.y,p.z, vs[i], c );
        }
        glEnd();
    }


    inline void simplex_deriv(
        const Vec2d& da, const Vec2d& db,
        double p7, double p8, double p9, double p4, double p5, double p6, double p2, double p3,
        Vec2d& deriv
    ){
        deriv.x = da.x*(p6-p4) + db.x*(p8-p2) + (da.x-db.x)*(p3-p7);
        deriv.y = da.y*(p6-p4) + db.y*(p8-p2) + (da.y-db.y)*(p3-p7);
    }

    void simplexGrid( int na, int nb, const Vec2d& da, const Vec2d& db,  const double * hs, const double * clrs, int ncolors, const uint32_t * cscale ){
        Vec2d pa; pa.set(0.0d);
        if( !cscale ){ cscale=&colors_rainbow[0]; ncolors=ncolors; }
        int ii=0;
        glNormal3f(0.0f,1.0f,0.0f);
        for (int ia=0; ia<(na-1); ia++){
            glBegin( GL_TRIANGLE_STRIP );
            Vec2d p; p.set(pa);
            for (int ib=0; ib<nb; ib++){
                double h=0.0d;
                //printf( " %i %i %i (%3.3f,%3.3f) %f %f \n", ia, ib, ii, p.x, p.y, hs[ii], clrs[ii] );
                if(clrs) colorScale( clrs[ii], ncolors, cscale );
                //if(hs){ simplex_deriv(); glNormal3f(0.0f,1.0f,0.0f); }
                if(hs){ h=hs[ii]; }
                glVertex3f( (float)(p.x), (float)(p.y), (float)h );

                if(clrs) colorScale( clrs[ii+nb], ncolors, cscale );
                //if(hs){ simplex_deriv(); glNormal3f(0.0f,1.0f,0.0f); }
                if(hs){ h=hs[ii+nb]; }
                glVertex3f( (float)(p.x+da.x), (float)(p.y+da.y), (float)h );
                p.add(db);
                ii++;
            }
            pa.add(da);
        glEnd();
        }

    }

    void simplexGridLines( int na, int nb, const Vec2d& da, const Vec2d& db,  const double * hs ){
        Vec2d p,pa; pa.set(0.0d);
        for (int ia=0; ia<(na-1); ia++){
            glBegin( GL_LINE_STRIP );
            p.set(pa);
            for (int ib=0; ib<nb; ib++){
                glVertex3f( (float)(p.x),      (float)(p.y),      (float)hs[ia*nb+ib] );
                p.add(db);
            }
            glEnd();
            p.set(pa);
            glBegin( GL_LINE_STRIP );
            for (int ib=0; ib<nb; ib++){
                int ii=ia*nb+ib;
                glVertex3f( (float)(p.x),      (float)(p.y),      (float)hs[ii   ] );
                glVertex3f( (float)(p.x+da.x), (float)(p.y+da.y), (float)hs[ii+nb] );
                p.add(db);
                ii++;
            }
            glEnd();
            pa.add(da);
        }
        p.set(pa);
        glBegin( GL_LINE_STRIP );
        for (int ib=0; ib<nb; ib++){
            glVertex3f( (float)(p.x),  (float)(p.y), (float)hs[(na-1)*nb+ib] );
            p.add(db);
        }
        glEnd();
    }

    void simplexGridLinesToned( int na, int nb, const Vec2d& da, const Vec2d& db,  const double * hs ){
        Vec2d p,pa; pa.set(0.0d);
        float h;
        for (int ia=0; ia<(na-1); ia++){
            glBegin( GL_LINE_STRIP );
            p.set(pa);
            for (int ib=0; ib<nb; ib++){
                h = (float)hs[ia*nb+ib];
                glColor3f( h,h*4,h*16 ); glVertex3f( (float)(p.x),      (float)(p.y),      h );
                p.add(db);
            }
            glEnd();
            p.set(pa);
            glBegin( GL_LINE_STRIP );
            for (int ib=0; ib<nb; ib++){
                int ii=ia*nb+ib;
                h=(float)hs[ii   ]; glColor3f( h,h*4,h*16 ); glVertex3f( (float)(p.x),      (float)(p.y),      h );
                h=(float)hs[ii+nb]; glColor3f( h,h*4,h*16 ); glVertex3f( (float)(p.x+da.x), (float)(p.y+da.y), h );
                p.add(db);
                ii++;
            }
            glEnd();
            pa.add(da);
        }
        p.set(pa);
        glBegin( GL_LINE_STRIP );
        for (int ib=0; ib<nb; ib++){
            h=(float)hs[(na-1)*nb+ib]; glColor3f( h,h*4,h*16 ); glVertex3f( (float)(p.x),  (float)(p.y), h );
            p.add(db);
        }
        glEnd();
    }


    void rectGridLines( Vec2i n, const Vec3d& p0, const Vec3d& da, const Vec3d& db ){
        glBegin( GL_LINES );
        Vec3d p  = p0;
        Vec3d dn = db*n.b;
        for (int ia=0; ia<n.a; ia++){
            glVertex3f( (float)(p .x), (float)(p .y), (float)(p .z) );  Vec3d p_ = p+dn;
            glVertex3f( (float)(p_.x), (float)(p_.y), (float)(p_.z) );
            //printf( "ia (%g,%g,%g) (%g,%g,%g)\n", p.x,p.y,p.z,   p_.x,p_.y,p_.z );
            p.add(da);
        }
        p   = p0;
        dn  = da*n.a;
        for (int ib=0; ib<n.b; ib++){
            glVertex3f( (float)(p .x), (float)(p .y), (float)(p .z) );  Vec3d p_ = p+dn;
            glVertex3f( (float)(p_.x), (float)(p_.y), (float)(p_.z) );
            //printf( "ib (%g,%g,%g) (%g,%g,%g)\n", p.x,p.y,p.z,   p_.x,p_.y,p_.z );
            p.add(db);
        }
        glEnd();
    }

    // int drawMesh( const Mesh& mesh  ){
    //     for( Polygon* pl : mesh.polygons ){
    //         Draw3D::drawPlanarPolygon( pl->ipoints.size(), &pl->ipoints.front(), &mesh.points.front() );
    //     }
    // }

    void text( const char * str, const Vec3f& pos, int fontTex, float textSize, int iend ){
        glDisable    ( GL_LIGHTING   );
        glDisable    ( GL_DEPTH_TEST );
        glShadeModel ( GL_FLAT       );
        glPushMatrix();
            //glMatrixMode(GL_MODELVIEW);
            //glMatrixMode(GL_PROJECTION);
            //printf("-- txt p (%f,%f,%f)\n", pos.x, pos.y, pos.z);
            glTranslatef( pos.x, pos.y, pos.z );
            //Draw::billboardCam( );
            billboardCamProj( );
            //Draw2D::drawString( inputText.c_str(), 0, 0, textSize, fontTex );
            text( str, fontTex, textSize, iend );
        glPopMatrix();
	};

    void text3D( const char * str, const Vec3f& pos, const Vec3f& fw, const Vec3f& up, int fontTex, float textSize, int iend ){
        glDisable    ( GL_LIGHTING   );
        glDisable    ( GL_DEPTH_TEST );
        glShadeModel ( GL_FLAT       );
        glPushMatrix();
            glTranslatef( pos.x, pos.y, pos.z );
            Mat3f rot;
            //rot.fromDirUp(fw,up);
            rot.fromSideUp(fw,up);
            float glmat[16];
            //toGLMatCam( {0.0,0.0,0.0},rot, glmat );
            toGLMat( {0.0,0.0,0.0},rot, glmat );
            glMatrixMode(GL_MODELVIEW);
            //glLoadMatrixf( glmat );
            glMultMatrixf( glmat );
            text( str, fontTex, textSize, iend );
        glPopMatrix();
    }

    void curve( float tmin, float tmax, int n, Func1d3 func ){
        glBegin(GL_LINE_STRIP);
        float dt = (tmax-tmin)/n;
        for( float t=tmin; t<=tmax; t+=dt ){
            double x,y,z;
            func( t, x, y, z );
            glVertex3f( (float)x, (float)y, (float)z );
        }
        glEnd();
    }

    void colorScale( int n, Vec3d pos, Vec3d dir, Vec3d up, void (_colorFunc_)(float f) ){
        glBegin(GL_TRIANGLE_STRIP);
        double d = 1.0/(n-1);
        for(int i=0; i<n; i++){
            double f = i*d;
            _colorFunc_( f );
            //glColor3f(1.0,1.0,1.0);
            Vec3d p = pos + dir*f;
            glVertex3f( (float)(p.x     ),(float)( p.y     ),(float)( p.z     ) );
            glVertex3f( (float)(p.x+up.x),(float)( p.y+up.y),(float)( p.z+up.z) );
            //printf( "(%g,%g,%g) (%g,%g,%g) \n", p.x, p.y, p.z, (float)(pos.x+up.x),(float)( pos.y+up.y),(float)( pos.z+up.z)  );
        }
        glEnd();
    }

// =================
// from drawUtils.h
// =================

void box( float x0, float x1, float y0, float y1, float z0, float z1, float r, float g, float b ){
    glBegin(GL_QUADS);
        glColor3f( r, g, b );
        glNormal3f(0,0,-1); glVertex3f( x0, y0, z0 ); glVertex3f( x1, y0, z0 ); glVertex3f( x1, y1, z0 ); glVertex3f( x0, y1, z0 );
        glNormal3f(0,-1,0); glVertex3f( x0, y0, z0 ); glVertex3f( x1, y0, z0 ); glVertex3f( x1, y0, z1 ); glVertex3f( x0, y0, z1 );
        glNormal3f(-1,0,0); glVertex3f( x0, y0, z0 ); glVertex3f( x0, y1, z0 ); glVertex3f( x0, y1, z1 ); glVertex3f( x0, y0, z1 );
        glNormal3f(0,0,+1); glVertex3f( x1, y1, z1 ); glVertex3f( x0, y1, z1 ); glVertex3f( x0, y0, z1 ); glVertex3f( x1, y0, z1 );
        glNormal3f(0,+1,1); glVertex3f( x1, y1, z1 ); glVertex3f( x0, y1, z1 ); glVertex3f( x0, y1, z0 ); glVertex3f( x1, y1, z0 );
        glNormal3f(+1,0,0); glVertex3f( x1, y1, z1 ); glVertex3f( x1, y0, z1 ); glVertex3f( x1, y0, z0 ); glVertex3f( x1, y1, z0 );
    glEnd();
}

void bbox( const Vec3f& p0, const Vec3f& p1 ){
    glBegin(GL_LINES);
        glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p0.y, p0.z );
        glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p0.x, p1.y, p0.z );
        glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p0.x, p0.y, p1.z );
        glVertex3f( p1.x, p1.y, p1.z ); glVertex3f( p0.x, p1.y, p1.z );
        glVertex3f( p1.x, p1.y, p1.z ); glVertex3f( p1.x, p0.y, p1.z );
        glVertex3f( p1.x, p1.y, p1.z ); glVertex3f( p1.x, p1.y, p0.z );
        glVertex3f( p1.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p0.z );
        glVertex3f( p1.x, p0.y, p0.z ); glVertex3f( p1.x, p0.y, p1.z );
        glVertex3f( p0.x, p1.y, p0.z ); glVertex3f( p1.x, p1.y, p0.z );
        glVertex3f( p0.x, p1.y, p0.z ); glVertex3f( p0.x, p1.y, p1.z );
        glVertex3f( p0.x, p0.y, p1.z ); glVertex3f( p1.x, p0.y, p1.z );
        glVertex3f( p0.x, p0.y, p1.z ); glVertex3f(p0.x, p1.y, p1.z );
    glEnd();
}

void triclinicBox( const Mat3f& lvec, const Vec3f& c0, const Vec3f& c1 ){
    Vec3f p0,p1;
    glBegin(GL_LINES);
        lvec.dot_to({c0.x,c0.y,c0.z},p0);
        lvec.dot_to({c0.x,c0.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to({c0.x,c1.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to({c1.x,c0.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to({c1.x,c1.y,c1.z},p0);
        lvec.dot_to({c0.x,c1.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to({c1.x,c0.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to({c1.x,c1.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to({c1.x,c0.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to({c1.x,c0.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to({c0.x,c0.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to({c0.x,c1.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to({c0.x,c1.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to({c1.x,c1.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
    glEnd();
}

void triclinicBoxT( const Mat3f& lvec, const Vec3f& c0, const Vec3f& c1 ){
    Vec3f p0,p1;
    glBegin(GL_LINES);
        lvec.dot_to_T({c0.x,c0.y,c0.z},p0);
        lvec.dot_to_T({c0.x,c0.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to_T({c0.x,c1.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to_T({c1.x,c0.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to_T({c1.x,c1.y,c1.z},p0);
        lvec.dot_to_T({c0.x,c1.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to_T({c1.x,c0.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        lvec.dot_to_T({c1.x,c1.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to_T({c1.x,c0.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to_T({c1.x,c0.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to_T({c0.x,c0.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to_T({c0.x,c1.y,c1.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to_T({c0.x,c1.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
        p0=p1; lvec.dot_to_T({c1.x,c1.y,c0.z},p1); glVertex3f( p0.x, p0.y, p0.z ); glVertex3f( p1.x, p1.y, p1.z );
    glEnd();
}


int boxList( float x0, float x1, float y0, float y1, float z0, float z1, float r, float g, float b  ){
    int ilist=glGenLists(1);
    glNewList( ilist, GL_COMPILE );
        box( x0, x1, y0, y1, z0, z1, r, g, b );
    glEndList();
    return( ilist );
    // don't forget use glDeleteLists( ilist ,1); later
}

void axis( float sc ){
    //glDisable (GL_LIGHTING);
    glBegin   (GL_LINES);
        glColor3f( 1, 0, 0 ); glVertex3f( 0, 0, 0 ); glVertex3f( 1*sc, 0, 0 );
        glColor3f( 0, 1, 0 ); glVertex3f( 0, 0, 0 ); glVertex3f( 0, 1*sc, 0 );
        glColor3f( 0, 0, 1 ); glVertex3f( 0, 0, 0 ); glVertex3f( 0, 0, 1*sc );
    glEnd();
}


}; // namespace Draw3D
