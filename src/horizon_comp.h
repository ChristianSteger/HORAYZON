#ifndef TESTLIB_H
#define TESTLIB_H

void horizon_comp(float* vert_grid, 
	int dem_dim_0, int dem_dim_1,
	float* vec_norm, float* vec_north,
	int offset_0, int offset_1,
	float* hori_buffer,
	int dim_in_0, int dim_in_1, int azim_num,
	float hori_acc, char* ray_algorithm, char* geom_type,
	float* vert_simp, int num_vert_simp,
	int* tri_ind_simp, int num_tri_simp,
    char* file_out,
    float* x_axis_val, float* y_axis_val,
    char* x_axis_name, char* y_axis_name, char* units,
    float hori_buffer_size_max,
    uint8_t* mask, float hori_fill);

#endif