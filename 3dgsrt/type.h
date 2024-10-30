#pragma once

#include "render/geometry.h"
#include "render/camera.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"

namespace Pupil::pt {

struct HitGroupData {
    optix::material::Material mat;
    optix::Geometry geo;
    int emitter_index_offset = -1;
};

struct OptixLaunchParams {
    struct {
        unsigned int max_depth;
        float scale_factor;
        bool accumulated_flag;

        struct {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;
    unsigned int random_seed;
    unsigned int sample_cnt;

    cuda::ConstDataView<optix::Camera> camera;
    optix::EmitterGroup emitters;

    cuda::RWArrayView<float4> accum_buffer;
    cuda::RWArrayView<float4> frame_buffer;

    cuda::RWArrayView<float3> normal_buffer;
    cuda::RWArrayView<float3> albedo_buffer;

    OptixTraversableHandle handle;
};

}// namespace Pupil::pt