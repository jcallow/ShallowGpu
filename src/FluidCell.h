/*
 * FluidCell.h
 *
 *  Created on: Jul 24, 2015
 *      Author: john
 */

#ifndef FLUIDCELL_H_
#define FLUIDCELL_H_

struct FluidCell {
	float u;
	float v;
	float d;
	FluidCell(float init_u, float init_v, float init_d ) : u(init_u), v(init_v), d(init_d) {}
};

#endif /* FLUIDCELL_H_ */
