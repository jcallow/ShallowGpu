/*
 * Grid.h
 *
 *  Created on: Jul 27, 2015
 *      Author: john
 */

#ifndef GRID_H_
#define GRID_H_

#include "FluidCell.h"
#include "Launch_Parameters.h"

class Grid {

public:
  int xdim;
  int ydim;
  FluidCell *advectionStep;
  FluidCell *finalStep;
  float *floor;
  float *height;
  float *advHeight;
  FluidCell *cells;

	Grid(int x, int y, float dt, float dx, float g, FluidCell *init_cells, float *init_floor );
	~Grid();

  __device__
  void advection(int x, int y);

  __device__
  void advection_u(int x, int y);

  __device__
  void advection_v(int x, int y);

  __device__
  void advection_d(int x, int y);

  __device__
  void h_adv(int x, int y);

  __device__
  void pressure_acceleration(int x, int y);

  __device__ __host__
  void depth_update(int x, int y);

  __device__
  void h_update(int x, int y);

	__device__
	void render(int x, int y, uchar4* bitmap);

	__device__
	void swap();

private:

  float delta_t;
  float delta_x;
  float gravity;
};

void start_Computation(uchar4* bitmap, Grid* ptr, Launch_Parameters* lp);

__global__
void Advection(Grid* ptr);

__global__
void pressure_acceleration(Grid* ptr);

__global__
void h_adv(Grid* ptr);

__global__
void depth_update(Grid* ptr);

__global__
void h_update(Grid* ptr);

__global__
void render(Grid* ptr, uchar4* bitmap);

__global__
void swap(Grid* ptr);



#endif /* GRID_H_ */
