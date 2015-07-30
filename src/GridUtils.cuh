
#ifndef GRIDUTILS_H_
#define GRIDUTILS_H_

#include "Launch_Parameters.h"
#include "Grid.cuh"
#include "Constants.h"

Grid* gridHostToDevice(Grid *ptr) {
  Grid *dev_ptr;
  cudaMalloc((void**)&dev_ptr, sizeof(Grid));
  cudaMemcpy(dev_ptr, ptr, sizeof(Grid), cudaMemcpyHostToDevice);

  FluidCell *hostCells;
  cudaMalloc((void **)&hostCells, ptr->xdim*ptr->ydim*sizeof(FluidCell));
  cudaMemcpy(hostCells, ptr->cells, ptr->xdim*ptr->ydim*sizeof(FluidCell), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dev_ptr->cells), &hostCells, sizeof(FluidCell *), cudaMemcpyHostToDevice);

  FluidCell *host_advection;
  cudaMalloc((void **)&host_advection, ptr->xdim*ptr->ydim*sizeof(FluidCell));
  cudaMemcpy(host_advection, ptr->advectionStep, ptr->xdim*ptr->ydim*sizeof(FluidCell), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dev_ptr->advectionStep), &host_advection, sizeof(FluidCell *), cudaMemcpyHostToDevice);

  FluidCell *host_temp;
  cudaMalloc((void **)&host_temp, ptr->xdim*ptr->ydim*sizeof(FluidCell));
  cudaMemcpy(host_temp, ptr->finalStep, ptr->xdim*ptr->ydim*sizeof(FluidCell), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dev_ptr->finalStep), &host_temp, sizeof(FluidCell *), cudaMemcpyHostToDevice);

  float *host_floor;
  cudaMalloc((void **)&host_floor, ptr->xdim*ptr->ydim*sizeof(float));
  cudaMemcpy(host_floor, ptr->floor, ptr->xdim*ptr->ydim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dev_ptr->floor), &host_floor, sizeof(float *), cudaMemcpyHostToDevice);

  float *host_height;
  cudaMalloc((void **)&host_height, ptr->xdim*ptr->ydim*sizeof(float));
  cudaMemcpy(host_height, ptr->height, ptr->xdim*ptr->ydim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dev_ptr->height), &host_height, sizeof(float *), cudaMemcpyHostToDevice);

  float *host_advHeight;
  cudaMalloc((void **)&host_advHeight, ptr->xdim*ptr->ydim*sizeof(float));
  cudaMemcpy(host_advHeight, ptr->advHeight, ptr->xdim*ptr->ydim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dev_ptr->advHeight), &host_advHeight, sizeof(float *), cudaMemcpyHostToDevice);

  return dev_ptr;
}

void gridDeviceToHost(Grid *d_ptr, Grid *ptr) {

}

Launch_Parameters* getDeviceLaunch(Grid* dev_ptr) {
  int xdim, ydim;
  cudaMemcpy(&xdim, &(dev_ptr->xdim), sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ydim, &(dev_ptr->ydim), sizeof(int), cudaMemcpyDeviceToHost);

  dim3 blocks(xdim/16, ydim/16);
  dim3 threads(16, 16);
  return new Launch_Parameters(blocks, threads);
}

Grid* initialize() {
  FluidCell *initial_cells = (FluidCell*)malloc(xDim*yDim*sizeof(FluidCell));
  float *floor = (float*)malloc(xDim*yDim*sizeof(float));

  for (int i = 0; i< xDim; i++) {
    for (int j = 0; j < yDim; j++) {
      float depth = (float)i/xDim;  //(1-(sqrt((xDim/2 - i)*(xDim/2 - i) + (yDim/2 - j)*(yDim/2 - j))/((yDim/2)*(sqrt(2)))))*0.7 + 0.1;
      float u = 0;
      float v = 0;
      initial_cells[i + j*xDim] = FluidCell(u,v,depth);
      floor[i] = 0;
    }
  }

  Grid* ptr_host = new Grid(xDim, yDim, delt, delx, gravity, initial_cells, floor);

  return ptr_host;
}

#endif
