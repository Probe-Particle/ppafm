#ifndef  integerOps_h
#define  integerOps_h

// ========= index poerations and modulo-math ===========

inline int wrap_index_fast( int i, int n){
    if(i<0){ return n+i; }else if (i>=n){ return i-n; };
    return i;
}

inline int clamp_index_fast( int i, int n){
    if(i<0){ return 0; }else if (i>=n){ return n-1; };
    return i;
}


inline int wrap_index2d_fast( int ix, int iy, int nx, int ny ){
    if(ix<0){ return nx+ix; }else if (ix>=nx){ return ix-nx; };
    if(iy<0){ return ny+iy; }else if (iy>=ny){ return iy-ny; };
    return iy*nx + ix;
}



// ========= Type conversion and packing ===========

inline uint32_t   pack32 (             uint16_t  x, uint16_t y  ){return x|(((uint32_t)y)<<16);};
inline void     unpack32 ( uint32_t i, uint16_t& x, uint16_t& y ){x=i&0xFFFF; y=(i>>16); };

inline uint32_t   pack32 (             uint8_t  x, uint8_t  y, uint8_t  z, uint8_t  w ){ return x|(((uint32_t)y)<<8)|(((uint32_t)z)<<16)|(((uint32_t)w)<<24);};
inline void     unpack32 ( uint32_t i, uint8_t& x, uint8_t& y, uint8_t& z, uint8_t& w ){ x=i&0xFF; y=((i&0xFF00)>>8); z=((i&0xFF0000)>>16); w=((i&0xFF000000)>>24); };

inline uint64_t   pack64 (             uint32_t  x, uint32_t  y ){return x|(((uint64_t)y)<<32);};
inline void     unpack64 ( uint64_t i, uint32_t& x, uint32_t& y ){x=i&0xFFFFFFFF; y=(i>>32);};

inline uint64_t   pack64 (             uint16_t  x, uint16_t  y, uint16_t  z, uint16_t  w ){return x|(((uint64_t)y)<<16)|(((uint64_t)z)<<32)|(((uint64_t)w)<<28);};
inline void     unpack64 ( uint64_t i, uint16_t& x, uint16_t& y, uint16_t& z, uint16_t& w ){x=i&0xFFFF; y=((i&0xFFFF0000)>>16); z=((i&0xFFFF00000000)>>32); w=((i&0xFFFF000000000000)>>48);};


// ========= Hashing ===========

// References:
// FNV hashes used in Bullet3 HashMap  http://www.isthe.com/chongo/tech/comp/fnv/

inline int rand_hash ( int r ){	return 1664525*r ^ 1013904223; }

inline int rand_hash2( int r ){
	r = 1664525*r ^ 1013904223;
	r = 1664525*r ^ 1013904223;
	return r;
}

// from http://burtleburtle.net/bob/hash/integer.html
inline uint32_t hash_Wang( uint32_t a){
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

inline unsigned int hash_Knuth( unsigned int i ){
	return ( i * 2654435761 >> 16 );
}

#endif
