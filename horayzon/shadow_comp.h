#include <embree3/rtcore.h>

namespace shapes {
class Rectangle {
public:
    int x0, y0, x1, y1;
    RTCDevice device;
    RTCScene scene;
    int dem_dim_0_cl, dem_dim_1_cl;
    float* vert_grid_cl;
    int offset_0_cl, offset_1_cl;
    float* vec_tilt_cl;
    float* vec_norm_cl;
    int dim_in_0_cl, dim_in_1_cl;
    float* surf_enl_fac_cl;
    Rectangle(int x0, int y0, int x1, int y1);
    ~Rectangle();
    int getArea();
    void move(int dx, int dy);
    void initialise(float* vert_grid,
    	int dem_dim_0, int dem_dim_1,
    	char* geom_type,
    	int offset_0, int offset_1,
    	float* vec_tilt,
    	float* vec_norm,
    	int dim_in_0, int dim_in_1,
    	float* surf_enl_fac);
    void shadow(float* sun_position, float* shaddow_buffer);
    void sw_dir_cor(float* sun_position, float* sw_dir_cor_buffer);
};
}