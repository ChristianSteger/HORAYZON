#include <embree4/rtcore.h>

namespace shapes {
class CppTerrain {
public:
    RTCDevice device;
    RTCScene scene;
    int dem_dim_0_cl, dem_dim_1_cl;
    float* vert_grid_cl;
    int offset_0_cl, offset_1_cl;
    float* vec_tilt_cl;
    float* vec_norm_cl;
    int dim_in_0_cl, dim_in_1_cl;
    float* surf_enl_fac_cl;
    float* elevation_cl;
    unsigned char* mask_cl;
    float sw_dir_cor_fill_cl;
    float ang_max_cl;
    int refrac_cor_cl;
    float temperature_ref_cl, pressure_ref_cl, lapse_rate_cl;
    float exp_cl;
    CppTerrain();
    ~CppTerrain();
    void initialise(float* vert_grid,
    	int dem_dim_0, int dem_dim_1,
    	int offset_0, int offset_1,
    	float* vec_tilt,
    	float* vec_norm,
    	int dim_in_0, int dim_in_1,
    	float* surf_enl_fac,
    	float* elevation,
    	unsigned char* mask,
    	char* geom_type,
    	float sw_dir_cor_fill,
    	float ang_max,
    	int refrac_cor);
    void shadow(float* sun_position, unsigned char* shadow_buffer);
    void sw_dir_cor(float* sun_position, float* sw_dir_cor_buffer);
};
}