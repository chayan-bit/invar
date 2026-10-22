/* invar-math-cr-v1 promises round-to-nearest-even, but CORE-MATH reads
   fegetround() and honours a directed mode left by the host process or a
   foreign call. Every call is bracketed by these: select round-to-nearest,
   then restore the caller's mode. */
#include <fenv.h>

int invar_math_hold_nearest(void) {
    int mode = fegetround();
    if (mode != FE_TONEAREST) {
        fesetround(FE_TONEAREST);
    }
    return mode;
}

void invar_math_restore_rounding(int mode) {
    if (mode != FE_TONEAREST) {
        fesetround(mode);
    }
}

/* Test support: the platform's upward mode and the current mode. */
int invar_math_fe_upward(void) { return FE_UPWARD; }
int invar_math_rounding(void) { return fegetround(); }
