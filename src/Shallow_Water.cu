#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "Grid.cuh"
#include "GridUtils.cuh"
#include "CudaMacroes.h"


#include "gpu_anim.h"
#include "Constants.h"


void generate_frame(uchar4* bitmap, Launch_Parameters* lp_ptr, Grid* device_ptr) {
  start_Computation(bitmap, device_ptr, lp_ptr);
}

int main (int argc, char* argv[]) {

  Grid* host_ptr = initialize();
  Grid* device_ptr = gridHostToDevice(host_ptr);
  Launch_Parameters* lp_ptr = getDeviceLaunch(device_ptr);

  GPUAnimBitmap  bitmap( xDim, yDim, device_ptr, lp_ptr);

  bitmap.anim_and_exit( (void (*)(uchar4*, Launch_Parameters*, Grid*))generate_frame, NULL );

  return 0;
}
