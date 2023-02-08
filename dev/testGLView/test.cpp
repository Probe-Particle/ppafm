
#include <cmath>
#include <cstdio>
#include "GLView.h"
#include "Draw3D.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>


void my_draw(){
    printf( "my_draw \n" );
    glClearColor( 0.5f, 0.5f, 0.5f, 1.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glEnable    ( GL_LIGHTING );
    glShadeModel( GL_FLAT     );

    Draw3D::box       ( -1.0f, 1.0f, -1.0f,   1.0f, -1.0f, 1.0f,   0.8f, 0.8f, 0.8f );

    glShadeModel( GL_SMOOTH     );

    glColor3f(1.0,0.5,0.7);
    Draw3D::sphere_oct( 5, 1.0f, (Vec3f){3.0,3.0,3.0} );

    glDisable ( GL_LIGHTING );
    Draw3D::axis ( 3.0f );
}

int main(){
    init( 640, 480 );
    set_draw_function( my_draw );
    run_Nframes(5000);

}
