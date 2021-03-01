#ifndef  GLView_h
#define  GLView_h

#include <stdbool.h> 

extern "C" {
typedef void (*ProcedurePointer)();

void init( int w, int h );
bool draw();
bool pre_draw();
bool post_draw();
void run_Nframes(int nframes);
void set_draw_function( ProcedurePointer draw_func );

} // extern "C" {

#endif






