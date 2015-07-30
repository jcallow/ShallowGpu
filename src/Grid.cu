/*
 * Grid.cpp
 *
 *  Created on: Jul 27, 2015
 *      Author: john
 */
#include "FluidCell.h"
#include "Grid.cuh"
#include <stdio.h>
#include <math.h>
#include "Launch_Parameters.h"


Grid::Grid(int x, int y, float dt, float dx, float g, FluidCell *init_cells, float *init_floor ) {
  delta_t = dt;
  delta_x = dx;
	xdim = x;
	ydim = y;
	gravity = g;

	cells = (FluidCell*)malloc(sizeof(FluidCell) * xdim*ydim);
	advectionStep = (FluidCell*)malloc(sizeof(FluidCell) * xdim*ydim);
	finalStep = (FluidCell*)malloc(sizeof(FluidCell) * xdim*ydim);
	floor = (float*)malloc(sizeof(float) * xdim*ydim);
	height = (float*)malloc(sizeof(float) * xdim*ydim);
	advHeight = (float*)malloc(sizeof(float) * xdim*ydim);

	for (int i = 0; i < xdim*ydim; i++) {
	  advectionStep[i] = FluidCell(0,0,0);
	  finalStep[i] = FluidCell(0,0,0);
	  height[i] = 0;
	  advHeight[i] = 0;
	  cells[i] = init_cells[i];
	  floor[i] = init_floor[i];
	}
}

Grid::~Grid() {

}

void start_Computation(uchar4* bitmap, Grid* ptr, Launch_Parameters* lp) {
  for (int i = 0; i < 5; i++) {
    Advection<<<lp->blocks, lp->threads>>>(ptr);
    cudaDeviceSynchronize();
    h_adv<<<lp->blocks, lp->threads>>>(ptr);
    cudaDeviceSynchronize();
    pressure_acceleration<<<lp->blocks, lp->threads>>>(ptr);
    cudaDeviceSynchronize();
    depth_update<<<lp->blocks, lp->threads>>>(ptr);
    cudaDeviceSynchronize();
    h_update<<<lp->blocks, lp->threads>>>(ptr);
    cudaDeviceSynchronize();
    swap<<<1,1>>>(ptr);
    cudaDeviceSynchronize();
  }
  render<<<lp->blocks, lp->threads>>>(ptr, bitmap);
  cudaDeviceSynchronize();
}

__global__
void Advection(Grid* ptr) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;

  ptr->advection(x,y);

}

__device__
void Grid::advection(int x, int y) {
  advection_u(x,y);
  advection_v(x,y);
  advection_d(x,y);
}

__device__
void Grid::advection_u(int x, int y) {
  int offset = x + y*xdim;
  int Fi, Fj, Ci, Cj, crd1, crd2, crd3, crd4;
  float Xi, Xj, a, b, c, d;

  if ((x>0) && (x<xdim-1)&&(y<ydim-1)) {
    Xi = x - (delta_t/delta_x)*cells[offset].u;
    Xj = y - (delta_t/delta_x)*(.25)*(cells[offset].v + cells[offset - 1].v + cells[offset + xdim].v + cells[offset - 1 + xdim].v);

    if ((Xi >= 0)&&(Xi<xdim-1)&&(Xj < ydim - 2)&&(Xj >= 0)) {
      Fi = std::floor(Xi);
      Ci = Fi + 1;
      Fj = std::floor(Xj);
      Cj = Fj + 1;

      // Find weights
      c = (Fi-Xi)*(Fj-Xj);
      d = (Xi-Fi)*(Cj-Xj);
      a = (Ci-Xi)*(Cj-Xj);
      b = (Ci-Xi)*(Xj-Fj);

      crd1 = Fi +Fj*xdim;
      crd2 = Fi +Cj*xdim;
      crd3 = Ci +Cj*xdim;
      crd4 = Ci +Fj*xdim;

      advectionStep[offset].u = a*cells[crd1].u + b*cells[crd2].u + c*cells[crd3].u + d*cells[crd4].u;
    }
    else {
      advectionStep[offset].u = 0;
    }
  }
  else {
    advectionStep[offset].u = 0;
  }
}

__device__
void Grid::advection_v(int x, int y) {
  int offset = x + y*xdim;
  int Fi, Fj, Ci, Cj, crd1, crd2, crd3, crd4;
  float Xi, Xj, a, b, c, d;

  if ((y>0) && (y<ydim-1) && (x < xdim-2)) {
    Xi = x - (delta_t/delta_x)*(0.25)*(cells[offset].u + cells[offset + 1].u + cells[offset - xdim].u + cells[offset + 1 - xdim].u);
    Xj = y - (delta_t/delta_x)*cells[offset].v;

    if ((Xi >= 0)&&(Xi<xdim-1)&&(Xj < ydim -1)&&(Xj >= 0)) {
      Fi = std::floor(Xi);
      Ci = Fi + 1;
      Fj = std::floor(Xj);
      Cj = Fj + 1;

      c = (Fi-Xi)*(Fj-Xj);
      d = (Xi-Fi)*(Cj-Xj);
      a = (Ci-Xi)*(Cj-Xj);
      b = (Ci-Xi)*(Xj-Fj);

      crd1 = Fi +Fj*xdim;
      crd2 = Fi +Cj*xdim;
      crd3 = Ci +Cj*xdim;
      crd4 = Ci +Fj*xdim;

      advectionStep[offset].v = a*advectionStep[crd1].v + b*advectionStep[crd2].v + c*advectionStep[crd3].v + d*advectionStep[crd4].v;
    }
    else {
      advectionStep[offset].v = 0;
    }
  }
  else {
    advectionStep[offset].v = 0;
  }
}

__device__
void Grid::advection_d(int x, int y) {
  int offset = x + y*xdim;
  int Fi, Fj, Ci, Cj, crd1, crd2, crd3, crd4;
  float Xi, Xj, a, b, c, d;

  if ((x<xdim-1)&&(y<ydim-1)){
    Xi = x - (delta_t/delta_x)*(cells[offset].u + cells[offset + 1].u);
    Xj = y - (delta_t/delta_x)*(cells[offset].v + cells[offset + xdim].v);

    if (Xi < 0) {
      Xi = 0;
    }
    else if (Xi >= xdim-1) {
      Xi = xdim-2;
    }

    if (Xj < 0) {
      Xj = 0;
    }
    else if (Xj >= ydim-1) {
      Xj = ydim-2;
    }

    if ((Xi >= 0)&&(Xi < xdim-1)&&(Xj >= 0)&&(Xj < ydim-1)) {
      Fi = std::floor(Xi);
      Ci = Fi + 1;
      Fj = std::floor(Xj);
      Cj = Fj + 1;

      c = (Fi-Xi)*(Fj-Xj);
      d = (Xi-Fi)*(Cj-Xj);
      a = (Ci-Xi)*(Cj-Xj);
      b = (Ci-Xi)*(Xj-Fj);

      crd1 = Fi +Fj*xdim;
      crd2 = Fi +Cj*xdim;
      crd3 = Ci +Cj*xdim;
      crd4 = Ci +Fj*xdim;

      advectionStep[offset].d = a*cells[crd1].d + b*cells[crd2].d + c*cells[crd3].d + d*cells[crd4].d;
    }
  }
  else {
    advectionStep[offset].d = 0;
  }
}

__global__
void pressure_acceleration(Grid* ptr) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;

  ptr->pressure_acceleration(x,y);
}

__device__
void Grid::pressure_acceleration(int x, int y) {
  int offset = x + y*xdim;
  int left = offset - 1;
  int down = offset - xdim;

  if ((x<xdim-1)&&(x>0)&&(y<ydim-1)) {
    finalStep[offset].u = advectionStep[offset].u - (delta_t/delta_x) * gravity * (advHeight[offset] - advHeight[left]);
  }

  if ((y>0)&&(y<ydim-1)&&(x<xdim-1)) {
    finalStep[offset].v = advectionStep[offset].v - (delta_t/delta_x) * gravity * (advHeight[offset] - advHeight[down]);
  }

  if ((x == 0) || (x == xdim -1)) {
    finalStep[offset].u = 0;
  }

  if ((y == 0) || (y == ydim -1)) {
    finalStep[offset].v = 0;
  }
}

__global__
void h_adv(Grid* ptr) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;

  ptr->h_adv(x, y);
}

__device__
void Grid::h_adv(int x, int y) {
  int offset = x + y*xdim;

  advHeight[offset] = floor[offset] + advectionStep[offset].d;
}

__global__
void depth_update(Grid* ptr) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;

  ptr->depth_update(x, y);
}

__device__
void Grid::depth_update(int x, int y) {
  int offset = x +y*xdim;
  int right = offset + 1;
  int up = offset + xdim;

  if ((x<xdim-1) && (y<ydim-1)) {
    finalStep[offset].d = advectionStep[offset].d - (delta_t/delta_x)*advectionStep[offset].d *
        (finalStep[right].u - finalStep[offset].u + finalStep[up].v - finalStep[offset].v);
  } else {
    finalStep[offset].d = advectionStep[offset].d;
  }
}

__global__
void h_update(Grid* ptr) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  ptr->h_update(x,y);
}

__device__
void Grid::h_update(int x, int y) {
  int offset = x + y*xdim;
  height[offset] = floor[offset] + finalStep[offset].d;
}

__global__
void swap(Grid* ptr) {
  ptr->swap();
}

__device__
void Grid::swap() {
  FluidCell* temp = cells;
  cells = finalStep;
  finalStep = temp;
}


__global__
void render(Grid* ptr, uchar4* bitmap) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  ptr->render(x, y, bitmap);
}

__device__
void Grid::render(int x, int y, uchar4* bitmap) {
  int offset = x + y*xdim;

  unsigned char blue = (unsigned char) 255 - (height[offset] * 256.0f);
  bitmap[offset].x = 0;
  bitmap[offset].y = 0;
  bitmap[offset].z = blue;
  bitmap[offset].w = 255;

}

