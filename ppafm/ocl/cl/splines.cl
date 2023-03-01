#ifndef splines_cl
#define splines_cl


float reciprocalInterp( float y1, float y2, float invW1, float invW2 ){
	// return   (y1/invW1   +  y2/invW2)/(1/invW1 + 1/invW2)    =    (y1*invW2 + y2*invW1)/(invW2 + invW1)
	return  (y1*invW2 + y2*invW1)/(invW2 + invW1);
}

float  spline_Hermite_y( float x,    float y0, float y1, float dy0, float dy1 ){
	float y01 = y0-y1;
	return      y0
		+x*(           dy0
		+x*( -3*y01 -2*dy0 - dy1
		+x*(  2*y01 +  dy0 + dy1 )));
}

float  spline_Hermite_dy( float x,    float y0, float y1, float dy0, float dy1 ){
	float y01 = y0-y1;
	return                 dy0
		+x*( 2*( -3*y01 -2*dy0 - dy1 )
		+x*  3*(  2*y01 +  dy0 + dy1 ));
}

float  spline_Hermite_ddy( float x, float y0, float y1, float dy0, float dy1 ){
	float y01 = y0-y1;
	return 2*( -3*y01 -2*dy0 - dy1 )
		+x*6*(  2*y01 +  dy0 + dy1 );
}

float4 spline_Hermite_basis( float x ){
    float4 c;
	float x2   = x*x;
	float K    =  x2*(x - 1);
	c.lo.s0        =  2*K - x2 + 1;   //    2*x3 - 3*x2 + 1
	c.lo.s1        = -2*K + x2    ;   //   -2*x3 + 3*x2
	c.hi.s0        =    K - x2 + x;   //      x3 - 2*x2 + x
	c.hi.s1        =    K         ;   //      x3 -   x2
    return c;
}

float4 spline_Hermite_dbasis( float x ){
    float4 c;
	float K    =  3*x*(x - 1);
	c.lo.s0        =  2*K        ;   //    6*x2 - 6*x
	c.lo.s1        = -2*K        ;   //   -6*x2 + 6*x
	c.hi.s0        =    K - x + 1;   //    3*x2 - 4*x + 1
	c.hi.s1        =    K + x    ;   //    3*x2 - 2*x
    return c;
}

float4 spline_Hermite_ddbasis( float x  ){
    float4 c;
//               x3     x2    x  1
	float x6   =  6*x;
	c.lo.s0        =  x6 + x6 -  6;   //    12*x - 6
	c.lo.s1        =   6 - x6 - x6;   //   -12*x + 6
	c.hi.s0        =  x6 -  4;        //     6*x - 4
	c.hi.s1        =  x6 -  2;        //     6*x - 2
    return c;
}


float spline_Hermite_y_arr( float x, __local float* ys  ){
	//val<T>( x, f1, f2, (f2-f0)*0.5, (f3-f1)*0.5 );
	float fi0;
	float dx = modf(x, &fi0 );
	int i0   = (int)fi0;
	//printf("x %g  dx %g i0 %g \n");
	float y0 = ys[i0  ];
	float y1 = ys[i0+1];
	float y2 = ys[i0+2];
	float y3 = ys[i0+3];
	float dy1 = (y2-y0) * 0.5;
	float dy2 = (y3-y1) * 0.5;
	return spline_Hermite_y( dx, y1, y2, dy1, dy2 );
}

/*
float spline_Hermite_dy_arr( float x, float* ys  ){
	//val<T>( x, f1, f2, (f2-f0)*0.5, (f3-f1)*0.5 );
	int i0;
	float dx = modf(x, &i0 );
	float y0 = ys[i0  ];
	float y1 = ys[i0+1];
	float y2 = ys[i0+2];
	float y3 = ys[i0+3];
	float dy1 = (y2-y0) * 0.5;
	float dy2 = (y3-y1) * 0.5;
	spline_Hermite_dy( dx, y1, y2, dy1, dy2 );
}

float spline_Hermite_ddy_arr( float x, float* ys  ){
	//val<T>( x, f1, f2, (f2-f0)*0.5, (f3-f1)*0.5 );
	int i0;
	float dx = modf(x, &i0 );
	float y0 = ys[i0  ];
	float y1 = ys[i0+1];
	float y2 = ys[i0+2];
	float y3 = ys[i0+3];
	float dy1 = (y2-y0) * 0.5;
	float dy2 = (y3-y1) * 0.5;
	spline_Hermite_ddy( dx, y1, y2, dy1, dy2 );
}
*/

#endif
