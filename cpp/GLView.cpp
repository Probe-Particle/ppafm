
#ifndef  GLView_cpp
#define  GLView_cpp

#include "Vec3.h"
#include "Vec2.h"
#include "Mat3.h"
#include "quaternion.h"

#include "Camera.h"
#include "Draw3D.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "GLView.h"  // THE HEADER

namespace Cam{

inline void ortho( const Camera& cam, bool zsym ){
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    float zmin = cam.zmin; if(zsym) zmin=-cam.zmax;
	glOrtho( -cam.zoom*cam.aspect, cam.zoom*cam.aspect, -cam.zoom, cam.zoom, zmin, cam.zmax );
	float glMat[16];
	Draw3D::toGLMatCam( { 0.0f, 0.0f, 0.0f}, cam.rot, glMat );
	glMultMatrixf( glMat );

	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity();
	glTranslatef(-cam.pos.x,-cam.pos.y,-cam.pos.z);
}

inline void perspective( const Camera& cam ){
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    //glFrustum( -ASPECT_RATIO, ASPECT_RATIO, -1, 1, camDist/zoom, VIEW_DEPTH );
    glFrustum( -cam.aspect*cam.zoom, cam.aspect*cam.zoom, -cam.zoom, cam.zoom, cam.zmin, cam.zmax );
    //glFrustum( -cam.zoom*cam.aspect, cam.zoom*cam.aspect, -cam.zoom, cam.zoom, cam.zmin, cam.zmax );
	float glMat[16];
	Draw3D::toGLMatCam( { 0.0f, 0.0f, 0.0f}, cam.rot, glMat );
	glMultMatrixf( glMat );

	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity();
	glTranslatef(-cam.pos.x,-cam.pos.y,-cam.pos.z);
    //glTranslatef ( -camPos.x+camMat.cx*camDist, -camPos.y+camMat.cy*camDist, -camPos.z+camMat.cz*camDist );
    //glTranslatef ( -cam.pos.x+camMat.cx*camDist, -camPos.y+camMat.cy*camDist, -camPos.z+camMat.cz*camDist );
}

}; // namespace Cam



void default_draw(){
    //printf( "default_draw \n" );
    glClearColor( 0.5f, 0.5f, 0.5f, 1.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glEnable    ( GL_LIGHTING );
    glShadeModel( GL_FLAT     );

    Draw3D::box       ( -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 0.8f, 0.8f, 0.8f );

    glShadeModel( GL_SMOOTH     );
    Draw3D::sphere_oct( 5, 1.0f, (Vec3f){3.0,3.0,3.0} );

    glDisable ( GL_LIGHTING );
    Draw3D::axis ( 3.0f );
}


class GLView{ public:

    constexpr static const float	VIEW_ZOOM_STEP       = 1.2f;
    constexpr static const float	VIEW_ZOOM_DEFAULT    = 10.0f;
    constexpr static const float	VIEW_DEPTH_DEFAULT   = 1000.0f;
    constexpr static const float	VIEW_MOVE_STEP       = 0.2f;

    int   WIDTH=800,HEIGHT=600;
    float VIEW_DEPTH=VIEW_DEPTH_DEFAULT;
    float ASPECT_RATIO=8.f/6.f;
    float zoom=10.f;

    //float camX0=0.0f,camY0=0.0f;
    //float fWIDTH, fHEIGHT, camXmin, camYmin, camXmax, camYmax;

    int   mouseX=0,mouseY=0;
    float mouse_begin_x=0;
    float mouse_begin_y=0;

    bool GL_LOCK = false;

    bool hasFocus;
    SDL_Window*      window;
    SDL_Renderer*    renderer;

    float  mouseRotSpeed   = 0.001;
    float  keyRotSpeed     = 0.01;
    float  cameraMoveSpeed = 0.2f;

    Quat4f qCamera    = Quat4fIdentity;
    Quat4f qCameraOld = Quat4fIdentity;
    //Mat3f  camMat     = Mat3fIdentity;

    bool  mouse_spinning    = false;

    Camera cam;

    float camDist = 50.0;
    Vec2i spinning_start;

    bool perspective  = false;
    bool first_person = false;


	public:
	bool LMB=false,RMB=false;
	int  upTime=0,delay=20,timeSlice=5,frameCount=0;
	bool loopEnd    = false, STOP = false;
	float camStep   = VIEW_MOVE_STEP;


// ============ function declarations

    void wait        (float ms);
    virtual void quit(       );
    void         wait(int ms);
    virtual int loop( int n );
    void defaultMouseHandling    ( const int& mouseX, const int& mouseY );

    // ==== Functions Virtual

    // ---- setup
    virtual void setupRenderer ();
    virtual void setDefaults   ();

    // ---- Inputs

    virtual void keyStateHandling( const Uint8 *keys       );
    virtual void eventHandling   ( const SDL_Event& event  );
    virtual void mouseHandling   (                         );
    virtual void inputHanding();
    virtual void updateMousePos ( int x, int y );

    // ---- Draw & Update
    virtual void camera      ();
    virtual void cameraHUD   ();
    virtual void draw        ();
    virtual void drawHUD     ();
    
    
    ProcedurePointer draw_func_ptr = default_draw;

    // ==== Functions

    // ---- Init
    void init( int& id, int WIDTH_, int HEIGHT_ );
    GLView ( int& id, int WIDTH_, int HEIGHT_ );

    // ---- Draw & Update
    void  startSpining ( float x, float y              ){ mouse_spinning = true; mouse_begin_x  = x; mouse_begin_y  = y;	}
    void  endSpining   (                               ){ mouse_spinning = false;	                                    }
    //void  projectMouse ( float mX, float mY, Vec3d& mp ){ mp.set_lincomb( mouseRight(mX), camRight,  mouseUp(mY), camUp ); };
    void  projectMouse ( float mX, float mY, Vec3f& mp ){ mp.set_lincomb( mouseRight(mX), cam.rot.a,  mouseUp(mY), cam.rot.b ); };
    void drawCrosshair( float sz );

    // ---- Camera
    //void camera_FPS       ( const Vec3d& pos, const Mat3d& rotMat );
    //void camera_FwUp      ( const Vec3d& pos, const Vec3d& fw, const Vec3d& up, bool upDominant );
    //void camera_FreeLook  ( const Vec3d& pos );
    //void camera_OrthoInset( const Vec2d& p1, const Vec2d& p2, const Vec2d& zrange, const Vec3d& fw, const Vec3d& up, bool upDominant );

    bool update( );
    bool pre_draw ( );
    bool post_draw( );

    inline float mouseUp     ( float mY ){ return 2*zoom*( 0.5 -mY/float(HEIGHT)                    ); };
    inline float mouseUp_    ( float mY ){ return 2*zoom*(      mY/float(HEIGHT) - 0.5              ); };
    inline float mouseRight  ( float mX ){ return 2*zoom*(      mX/float(HEIGHT) - 0.5*ASPECT_RATIO ); };

    inline bool wait_LOCK( int n, int ms ){ if(!GL_LOCK) return true; for(int i=0; i<n; i++){ SDL_Delay(ms); if(!GL_LOCK) return true; } return false; }

};


// ================================
// ======== Implementation 
// =================================

// ===================== INIT


void GLView::setupRenderer(){
    //float white    [] = { 1.0f, 1.0f,  1.0f,  1.0f };
    float ambient  [] = { 0.1f, 0.15f, 0.25f, 1.0f };
    float diffuse  [] = { 0.9f, 0.8f,  0.7f,  1.0f };
    float specular [] = { 1.0f, 1.0f,  1.0f,  1.0f };
    float shininess[] = { 80.0f                    };
    float lightPos [] = { 1.0f, -1.0f, 1.0f, 0.0f  };

    //glMaterialfv ( GL_FRONT_AND_BACK, GL_AMBIENT,   ambient);
    //glMaterialfv ( GL_FRONT_AND_BACK, GL_DIFFUSE,   diffuse);
    glMaterialfv ( GL_FRONT_AND_BACK, GL_SPECULAR,  specular);
    glMaterialfv ( GL_FRONT_AND_BACK, GL_SHININESS, shininess);

    glEnable     ( GL_COLOR_MATERIAL    );
    glLightfv    ( GL_LIGHT0, GL_POSITION,  lightPos );
    glLightfv    ( GL_LIGHT0, GL_DIFFUSE,   diffuse  );
    glLightfv    ( GL_LIGHT0, GL_AMBIENT,   ambient  );
    glLightfv    ( GL_LIGHT0, GL_SPECULAR,  specular );
    //glLightfv    ( GL_LIGHT0, GL_AMBIENT,  ambient  );
    glEnable     ( GL_LIGHTING         );
    glEnable     ( GL_LIGHT0           );
    glEnable     ( GL_NORMALIZE        );

    //glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, 1 );

    glEnable     ( GL_DEPTH_TEST       );
    glHint       ( GL_LINE_SMOOTH_HINT, GL_NICEST );
    glShadeModel ( GL_SMOOTH           );
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

void GLView::setDefaults(){
    VIEW_DEPTH   = VIEW_DEPTH_DEFAULT;
    ASPECT_RATIO = WIDTH/(float)HEIGHT;
    zoom         = VIEW_ZOOM_DEFAULT;
    //printf(" %f %f %f \n", zoom, ASPECT_RATIO, VIEW_DEPTH  );
    mouse_begin_x  = 0;
    mouse_begin_y  = 0;
}

void GLView::init( int& id, int WIDTH_, int HEIGHT_ ){
    WIDTH  = WIDTH_;
    HEIGHT = HEIGHT_;
    setDefaults();
    SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, SDL_WINDOW_OPENGL, &window, &renderer);
    id = SDL_GetWindowID(window); printf( " win id %i \n", id );
    char str[40];  sprintf(str, " Window id = %d", id );
    SDL_SetWindowTitle( window, str );
    setupRenderer();
    //printf( " ASPECT_RATIO %f \n", ASPECT_RATIO );
}

GLView::GLView( int& id, int WIDTH_, int HEIGHT_ ){
    init(id,WIDTH_,HEIGHT_);
    qCamera.setOne();
    qCamera.toMatrix_unitary( cam.rot );
    cam.pos.set(0.0d);
    GLbyte* s;
    // http://stackoverflow.com/questions/40444046/c-how-to-detect-graphics-card-model
    printf( "GL_VENDOR  : %s \n", glGetString(GL_VENDOR)  );
    printf( "GL_VERSION : %s \n", glGetString(GL_VERSION) );
}

// ===================== UPDATE & DRAW

/*
bool GLView::update( ){
    //SDL_RenderPresent(renderer);
    //glPushMatrix();
    if( GL_LOCK ){ printf("ScreenSDL2OGL::update GL_LOCK\n"); return; }
    GL_LOCK = true;
    camera();
    draw();
    cameraHUD();
    drawHUD();
    //glPopMatrix();
    //glFlush();
    SDL_RenderPresent(renderer);
    GL_LOCK = false;
}
*/

bool GLView::update( ){
    //printf( "GLView::update %i \n", GL_LOCK );
    if( GL_LOCK ) return true;
    pre_draw();
    draw();
    cameraHUD();
    drawHUD();
    post_draw();
    return GL_LOCK;
}

bool GLView::pre_draw(){
    //thisApp->update();
    if( GL_LOCK ) return true;
    inputHanding();
    GL_LOCK = true;
    camera();

    glClearColor( 0.5f, 0.5f, 0.5f, 1.0f );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glEnable    ( GL_LIGHTING );
	glShadeModel( GL_FLAT     );

    //printf( "DEBUG pre_draw[fame=%i] \n", frameCount );
    return GL_LOCK;
}

bool GLView::post_draw(){
    //printf( "DEBUG post_draw[fame=%i] \n", frameCount );
    SDL_RenderPresent(renderer);
    frameCount++;
    GL_LOCK = false;
    return loopEnd;
}

// void GLView::camera(){
//     glMatrixMode( GL_PROJECTION );
//     glLoadIdentity();
//     glOrtho ( -zoom*ASPECT_RATIO, zoom*ASPECT_RATIO, -zoom, zoom, -VIEW_DEPTH, +VIEW_DEPTH );
//     glTranslatef( -camX0, -camY0, 0.0f );
//     glMatrixMode (GL_MODELVIEW);
// }

void GLView::camera(){
    ((Quat4f)qCamera).toMatrix(cam.rot);
    cam.zoom   = zoom;
    cam.aspect = ASPECT_RATIO;
    if (perspective){ Cam::perspective( cam ); }
    else            { Cam::ortho( cam, true ); }
}

void GLView::cameraHUD(){
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho ( 0, WIDTH, 0, HEIGHT, -VIEW_DEPTH, +VIEW_DEPTH );
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity();
}

void GLView::updateMousePos ( int x, int y ){
    mouse_begin_x = mouseRight( x );
    mouse_begin_y = mouseUp   ( y );
}

//void GLView::draw   (){
//    glClearColor( 0.5f, 0.5f, 0.5f, 0.0f );
//    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//}

void GLView::drawHUD(){ 
}

void GLView::draw   (){
    //printf( "GLView::draw \n" );
    draw_func_ptr();

/*
    glClearColor( 0.5f, 0.5f, 0.5f, 1.0f );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glEnable    ( GL_LIGHTING );
	glShadeModel( GL_FLAT     );

	Draw3D::box       ( -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 0.8f, 0.8f, 0.8f );

	glShadeModel( GL_SMOOTH     );
	Draw3D::sphere_oct( 5, 1.0f, (Vec3f){3.0,3.0,3.0} );

	glDisable ( GL_LIGHTING );
	Draw3D::axis ( 3.0f );
*/
}


// ===================== INPUTS

void GLView::quit(){ SDL_Quit(); loopEnd=true; /*exit(1);*/ };

void GLView::wait(int ms){
    for( int i=0; i<ms; i+=timeSlice ){
        uint32_t tnow=SDL_GetTicks();
        if(tnow>=(upTime+ms)){upTime=tnow; break; }
        SDL_Delay(timeSlice);
    }
}

int GLView::loop( int n ){
    loopEnd = false;
    int iframe=0;
    for( ; iframe<n; iframe++ ){
        //printf( "GLView::loop[%i] \n", iframe );
        inputHanding();
        //if(!STOP){update();} // DEPRECATED: usually we want to stop physics, not drawing
        update();
        //printf(" %i \n", iframe );
        wait(delay);
        //if( delay>0 ) SDL_Delay( delay );
        if(loopEnd) break;
    }
    return iframe;
}

void GLView::inputHanding(){
    const Uint8 *keys = SDL_GetKeyboardState(NULL);
	keyStateHandling ( keys );
	mouseHandling( );
    SDL_Event		 event;
	while(SDL_PollEvent(&event)){
	    eventHandling( event );
	}
}

void GLView::eventHandling ( const SDL_Event& event  ){
    switch( event.type ){
        case SDL_KEYDOWN :
            switch( event.key.keysym.sym ){
                case SDLK_ESCAPE:   quit(); break;
                //case SDLK_SPACE:    STOP = !STOP; printf( STOP ? " STOPED\n" : " UNSTOPED\n"); break;
                case SDLK_KP_MINUS: zoom*=VIEW_ZOOM_STEP; break;
                case SDLK_KP_PLUS:  zoom/=VIEW_ZOOM_STEP; break;
                case SDLK_o:  perspective   = !perspective; break;
                case SDLK_p:  first_person  = !first_person ;   break;
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            switch( event.button.button ){
                case SDL_BUTTON_LEFT:  LMB = true;  break;
                case SDL_BUTTON_RIGHT: RMB = true;  break;
            };  break;
        case SDL_MOUSEBUTTONUP:
            switch( event.button.button ){
                case SDL_BUTTON_LEFT:  LMB = false; break;
                case SDL_BUTTON_RIGHT: RMB = false; break;
            }; break;
        case SDL_QUIT: quit(); break;
    };
}

void GLView::keyStateHandling( const Uint8 *keys ){

    if( keys[ SDL_SCANCODE_LEFT  ] ){ qCamera.dyaw  (  keyRotSpeed ); }
    if( keys[ SDL_SCANCODE_RIGHT ] ){ qCamera.dyaw  ( -keyRotSpeed ); }
    if( keys[ SDL_SCANCODE_UP    ] ){ qCamera.dpitch(  keyRotSpeed ); }
    if( keys[ SDL_SCANCODE_DOWN  ] ){ qCamera.dpitch( -keyRotSpeed ); }

    if( keys[ SDL_SCANCODE_A ] ){ cam.pos.add_mul( cam.rot.a, -cameraMoveSpeed ); }
    if( keys[ SDL_SCANCODE_D ] ){ cam.pos.add_mul( cam.rot.a,  cameraMoveSpeed ); }
    if( keys[ SDL_SCANCODE_W ] ){ cam.pos.add_mul( cam.rot.b,  cameraMoveSpeed ); }
    if( keys[ SDL_SCANCODE_S ] ){ cam.pos.add_mul( cam.rot.b, -cameraMoveSpeed ); }
    if( keys[ SDL_SCANCODE_Q ] ){ cam.pos.add_mul( cam.rot.c, -cameraMoveSpeed ); }
    if( keys[ SDL_SCANCODE_E ] ){ cam.pos.add_mul( cam.rot.c,  cameraMoveSpeed ); }

    //printf( "frame %i keyStateHandling cam.pos (%g,%g,%g) \n", frameCount, cam.pos.x, cam.pos.y, cam.pos.z );

}


void GLView::mouseHandling( ){
    SDL_GetMouseState( &mouseX, &mouseY );   mouseY=HEIGHT-mouseY;
    mouse_begin_x = (2*mouseX-WIDTH )*zoom/HEIGHT;
    mouse_begin_y = (2*mouseY-HEIGHT)*zoom/HEIGHT;
    int mx,my; Uint32 buttons = SDL_GetRelativeMouseState( &mx, &my);
    //printf( " %i %i \n", mx,my );
    if ( buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
        Quat4f q; q.fromTrackball( 0, 0, -mx*mouseRotSpeed, my*mouseRotSpeed );
        qCamera.qmul_T( q );
    }
    //qCamera.qmul( q );
}

void GLView::drawCrosshair( float sz ){
    glBegin( GL_LINES );
    float whalf = WIDTH *0.5;
    float hhalf = HEIGHT*0.5;
    glVertex3f( whalf-10,hhalf, 0 ); glVertex3f( whalf+10,hhalf, 0 );
    glVertex3f( whalf,hhalf-10, 0 ); glVertex3f( whalf,hhalf+10, 0 );
    glEnd();
}



/*
// ============= CAMERAS


void GLView::camera(){
    //printf( "ScreenSDL2OGL_3D::camera() \n" );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho ( -zoom*ASPECT_RATIO, zoom*ASPECT_RATIO, -zoom, zoom, -VIEW_DEPTH, +VIEW_DEPTH );
    //glOrtho ( -zoom, zoom, -zoom, zoom, -VIEW_DEPTH, +VIEW_DEPTH );
    //glOrtho ( -zoom, zoom, -zoom*ASPECT_RATIO, zoom*ASPECT_RATIO, -VIEW_DEPTH, +VIEW_DEPTH );
    glMatrixMode (GL_MODELVIEW);
    //float camMatrix[4][4];
    //build_rotmatrix (camMatrix, qCamera );
    //glLoadMatrixf(&camMatrix[0][0]);
}

void GLView::camera_FPS( const Vec3d& pos, const Mat3d& rotMat ){
    //glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum( -ASPECT_RATIO, ASPECT_RATIO, -1, 1, camDist/zoom, VIEW_DEPTH );
    //Mat3d camMat;
    Vec3f camPos;
    convert( pos, cam.pos );
    //cam.rot.setT( (Mat3f)rotMat );
    cam.rot.set( (Mat3f)rotMat );
    float glMat[16];
    Draw3D::toGLMatCam( { 0.0f, 0.0f, 0.0f}, cam.rot, glMat );
    glMultMatrixf( glMat );
    //glTranslatef ( -camPos.x+camMat.cx*camDist, -camPos.y+camMat.cy*camDist, -camPos.z+camMat.cz*camDist );
    glTranslatef ( -cam.pos.x+cam.rot.cx*camDist, -cam.pos.y+cam.rot.cy*camDist, -cam.pos.z+cam.rot.cz*camDist );
};

// camera( pos, dir, Up )
void GLView::camera_FwUp( const Vec3d& pos, const Vec3d& fw, const Vec3d& up, bool upDominant ){
    //glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum( -ASPECT_RATIO, ASPECT_RATIO, -1, 1, camDist/zoom, VIEW_DEPTH );
    //Mat3d camMat;
    //Vec3f camPos;
    convert( pos, cam.pos );
    cam.rot.b = (Vec3f)up;
    cam.rot.c = (Vec3f)fw;
    if( upDominant ){
        cam.rot.b.normalize();
        cam.rot.c.makeOrtho( cam.rot.b );
        cam.rot.c.normalize();
    }else{
        cam.rot.c.normalize();
        cam.rot.b.makeOrtho( cam.rot.c );
        cam.rot.b.normalize();
    }
    cam.rot.a.set_cross(cam.rot.b,cam.rot.c);
	float glMat[16];
	Draw3D::toGLMatCam( { 0.0f, 0.0f, 0.0f}, cam.rot, glMat );
	glMultMatrixf( glMat );
    //glTranslatef ( -camPos.x+camMat.cx*camDist, -camPos.y+camMat.cy*camDist, -camPos.z+camMat.cz*camDist );
    glTranslatef ( -cam.pos.x+cam.rot.cx*camDist, -cam.pos.y+cam.rot.cy*camDist, -cam.pos.z+cam.rot.cz*camDist );
};

void GLView::camera_FreeLook( const Vec3d& pos ){
    //glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum( -ASPECT_RATIO, ASPECT_RATIO, -1, 1, camDist/zoom, VIEW_DEPTH );
    //Mat3d camMat;
    //Vec3f camPos;
    convert( pos, cam.pos );
    qCamera.toMatrix( cam.rot );
    cam.rot.makeT();
	float glMat[16];
	Draw3D::toGLMatCam( { 0.0f, 0.0f, 0.0f}, cam.rot, glMat );
	glMultMatrixf( glMat );
    //glTranslatef ( -camPos.x+camMat.cx*camDist, -camPos.y+camMat.cy*camDist, -camPos.z+camMat.cz*camDist );
    glTranslatef ( -cam.pos.x+cam.rot.cx*camDist, -cam.pos.y+cam.rot.cy*camDist, -cam.pos.z+cam.rot.cz*camDist );
};

void GLView::camera_OrthoInset( const Vec2d& p1, const Vec2d& p2, const Vec2d& zrange, const Vec3d& fw, const Vec3d& up, bool upDominant ){
    //glMatrixMode( GL_PROJECTION ); glPushMatrix();
    glLoadIdentity();
    //glOrtho( -ASPECT_RATIO*5.0, ASPECT_RATIO*30.0, -5.0, 30.0,  -100.0, 100.0);
    //printf( "--- %f %f  %f %f  %f %f \n", -ASPECT_RATIO*5.0, ASPECT_RATIO*30.0, -5.0, 30.0,  -100.0, 100.0  );
    //printf( "    %f %f  %f %f  %f %f \n", ASPECT_RATIO*p1.x, ASPECT_RATIO*p2.x, p1.y, p2.y,   zrange.a, zrange.b );
    glOrtho( ASPECT_RATIO*p1.x, ASPECT_RATIO*p2.x, p1.y, p2.y, zrange.a, zrange.b );
    //Mat3d camMat;
    cam.rot.b = (Vec3f)up;
    cam.rot.c = (Vec3f)fw;
    if( upDominant ){
        cam.rot.b.normalize();
        cam.rot.c.makeOrtho( cam.rot.b );
        cam.rot.c.normalize();
    }else{
        cam.rot.c.normalize();
        cam.rot.b.makeOrtho( cam.rot.c );
        cam.rot.b.normalize();
    }
    cam.rot.a.set_cross(cam.rot.b,cam.rot.c);
    float glMat[16];
    Draw3D::toGLMatCam( {0.0f,0.0f,0.0f}, cam.rot, glMat );
    //Draw3D::toGLMat( { 0.0f, 0.0f, 0.0f}, camMat, glMat );
    glMultMatrixf( glMat );
    //glMatrixMode (GL_MODELVIEW);
}

void GLView::camera(){
    ((Quat4f)qCamera).toMatrix(cam.rot);
    cam.zoom   = zoom;
    cam.aspect = ASPECT_RATIO;
    //Cam::ortho( cam, true );
    //Cam::perspective( cam );
    if (perspective){ Cam::perspective( cam ); }
    else            { Cam::ortho( cam, true ); }

}
*/






GLView * thisApp = 0;

extern "C"{

//void init( int w, int h, void* world_, char* work_dir_ ){
void init( int w, int h ){
    //world = (FlightWorld*)world_;
    //work_dir = work_dir_;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
    //SDL_SetRelativeMouseMode( SDL_TRUE );
    //SDL_ShowCursor(SDL_DISABLE);
    int junk;
    thisApp = new GLView( junk , w, h );
    SDL_SetWindowPosition(thisApp->window, 100, 0 );
    thisApp->loopEnd = false;
}

bool draw(){
   return thisApp->update();
}

bool pre_draw(){
    return thisApp->pre_draw();
}

bool post_draw(){
    return thisApp->post_draw();
}

void run_Nframes(int nframes){
    thisApp->loop( nframes );
}

void set_draw_function( ProcedurePointer draw_func ){
    thisApp->draw_func_ptr = draw_func;
}

} // extern "C"{





#endif






