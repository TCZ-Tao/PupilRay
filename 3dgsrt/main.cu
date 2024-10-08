#include <optix.h>
#include "type.h"

#include "optix/util.h"
#include "render/geometry.h"
#include "render/material/bsdf/bsdf.h"

#include "cuda/random.h"
#include "resource/3dgs/ply_loader.h"

using namespace Pupil;


extern "C" {
__constant__ pt::OptixLaunchParams optix_launch_params;
}

// kbuffer in PRD
struct ThreedgsQueue{
    unsigned int hit_count;
    float        hit_t[PLY_3DGS_CHUNK_SIZE];
    unsigned int hit_index[PLY_3DGS_CHUNK_SIZE];
    void*        sbts[PLY_3DGS_CHUNK_SIZE];
};

struct HitInfo {
    optix::LocalGeometry geo;
    optix::material::Material::LocalBsdf bsdf;
    int emitter_index;
};

struct PathPayloadRecord {
    float3 radiance;
    float3 env_radiance;
    float env_pdf;
    cuda::Random random;

    float3 throughput;

    HitInfo hit;

    ThreedgsQueue queue;

    unsigned int depth;
    bool done;

    float test;
};

__forceinline__ __device__ float3 GetSHCoef(pt::HitGroupData* data,
    const unsigned int index_begin,
    const unsigned int index) { 
    const unsigned int sh_n = 16;
    if (index == 0) {
        return make_float3(data->geo.threedgs.shses[index_begin],
                    data->geo.threedgs.shses[index_begin+1],
                    data->geo.threedgs.shses[index_begin+2]);
    } else {
        return make_float3(data->geo.threedgs.shses[index_begin + index - 1 + 3],
                    data->geo.threedgs.shses[index_begin + index - 1 + sh_n + 2],
                    data->geo.threedgs.shses[index_begin + index - 1 + sh_n * 2 + 1]);
    }
}
__forceinline__ __device__ float3 ComputeSH(pt::HitGroupData* data,
    const unsigned int index_begin, const float x, const float y, const float z) {
    float3 c = GetSHCoef(data, index_begin, 0) * 0.28209479177387814f;

    float  x2 = x * x;
    float  y2 = y * y;
    float  z2 = z * z;

    c -= 0.4886025119029199f * y * GetSHCoef(data, index_begin, 1); 
    c += 0.4886025119029199f * z * GetSHCoef(data, index_begin, 2); 
    c -= 0.4886025119029199f * x * GetSHCoef(data, index_begin, 3); 
                                                          
    c += 1.0925484305920792f  * x * y *                GetSHCoef(data, index_begin, 4); 
    c += -1.0925484305920792f * y * z *                GetSHCoef(data, index_begin, 5); 
    c += 0.31539156525252005f * (2.f * z2 - x2 - y2) * GetSHCoef(data, index_begin, 6); 
    c += -1.0925484305920792f * z * x *                GetSHCoef(data, index_begin, 7); 
    c += 0.5462742152960396f  * (x2 - y2) *            GetSHCoef(data, index_begin, 8); 

    c += -0.5900435899266435f * (3.f * x2 - y2) * y *                GetSHCoef(data, index_begin, 9); 
    c += 2.890611442640554f   * x * y * z *                          GetSHCoef(data, index_begin, 10); 
    c += -0.4570457994644658f * (4.f * z2 - x2 - y2) * y *           GetSHCoef(data, index_begin, 11); 
    c += 0.3731763325901154f  * z * (2.f * z2 - 3.f * x2 - 3.f * y2)*GetSHCoef(data, index_begin, 12); 
    c += -0.4570457994644658f * x * (4.f * z2 - x2 - y2) *           GetSHCoef(data, index_begin, 13); 
    c += 1.445305721320277f   * (x2 - y2) * z *                      GetSHCoef(data, index_begin, 14); 
    c += -0.5900435899266435f * x * (x2 - 3.f * y2) *                GetSHCoef(data, index_begin, 15); 

    c += 0.5f;
    if (c.x < 0.f) c.x = 0.f;
    return c;
}

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    auto &camera = *optix_launch_params.camera.GetDataPtr();

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix::PackPointer(&record, u0, u1);

    record.done = false;
    record.depth = 0u;
    record.throughput = make_float3(1.f);
    record.radiance = make_float3(0.f);
    record.env_radiance = make_float3(0.f);
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);
    record.test = 0.f;

    const float2 subpixel = make_float2((static_cast<float>(index.x)) / w,
                                        (static_cast<float>(index.y)) / h);
    const float4 point_on_film = make_float4(subpixel, 0.f, 1.f);

    float4 d = camera.sample_to_camera * point_on_film;

    d /= d.w;
    d.w = 0.f;
    d = normalize(d);

    float3 ray_direction = normalize(make_float3(camera.camera_to_world * d));

    float3 ray_origin = make_float3(
        camera.camera_to_world.r0.w,
        camera.camera_to_world.r1.w,
        camera.camera_to_world.r2.w);

    optix_launch_params.albedo_buffer[pixel_index] = make_float3(0.f);
    optix_launch_params.normal_buffer[pixel_index] = make_float3(0.f);
    optix_launch_params.accum_buffer[pixel_index] = make_float4(0.f);
    
    float t_min = 0.001f;
    float t_max = 1e16f; //t_scene_max?

    //optixTrace(optix_launch_params.handle,
    //           ray_origin, ray_direction,
    //           t_min, t_max, 0.f,
    //           255, OPTIX_RAY_FLAG_NONE,
    //           0, 2, 0,
    //           u0, u1);
    //auto local_hit = record.hit;

    //if (record.done) {
    //    optix_launch_params.frame_buffer[pixel_index] = make_float4(1.f); // background
    //} else {
    //    optix_launch_params.frame_buffer[pixel_index] = 
    //        make_float4(record.hit.geo.texcoord.x, record.hit.geo.texcoord.y, 
    //            1.f - record.hit.geo.texcoord.x - record.hit.geo.texcoord.y, 1.f);
    //}

    float  t_curr = t_min;
    float3 radiance = make_float3(0.f);
    float  transmittance = 1.f;
    float  transmittance_min = 0.01f;

    auto ray_count = 0u;

    while (t_curr < t_max && !record.done && transmittance > transmittance_min) {
        // reset kbuffer
        record.queue.hit_count = 0;
        for (int i = 0; i < PLY_3DGS_CHUNK_SIZE; ++i)
            record.queue.hit_t[i] = t_max;

        optixTrace(optix_launch_params.handle,
            ray_origin, ray_direction,
            t_curr, t_max, 0.f,
            255, OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            0, 2, 0,
            u0, u1);

        unsigned int hit_count = record.queue.hit_count;
        const unsigned int queue_size = min(PLY_3DGS_CHUNK_SIZE, hit_count);
        for (int i = 0; i < queue_size; ++i) {
            const unsigned int particle_index = record.queue.hit_index[i];
            pt::HitGroupData* data = static_cast<pt::HitGroupData*>(record.queue.sbts[i]);
            // compute response
            float3 scale = data->geo.threedgs.scales[particle_index] *
                optix_launch_params.config.scale_factor;
            float3 scale_inv = 1.f / scale;
            float4 q = data->geo.threedgs.rotations[particle_index];
            float3 r0 = make_float3( //不知道是转置还是不转的,试试
                1.f - 2.f * q.y * q.y - 2.f * q.z * q.z,
                2.f * q.x * q.y - 2.f * q.z * q.w,
                2.f * q.x * q.z + 2.f * q.y * q.w);
            float3 r1 = make_float3(
                2.f * q.x * q.y + 2.f * q.z * q.w,
                1.f - 2.f * q.x * q.x - 2.f * q.z * q.z,
                2.f * q.y * q.z - 2.f * q.x * q.w);
            float3 r2 = make_float3(
                2.f * q.x * q.z - 2.f * q.y * q.w,
                2.f * q.y * q.z + 2.f * q.x * q.w,
                1.f - 2.f * q.x * q.x - 2.f * q.y * q.y);
            float3 rt0 = make_float3(r0.x, r1.x, r2.x);
            float3 rt1 = make_float3(r0.y, r1.y, r2.y);
            float3 rt2 = make_float3(r0.z, r1.z, r2.z);
            // S^{-1}*R^T
            float4 m0 = make_float4( 
                scale_inv.x * rt0.x, scale_inv.x * rt0.y, scale_inv.x * rt0.z, 0.f);
            float4 m1 = make_float4( 
                scale_inv.y * rt1.x, scale_inv.y * rt1.y, scale_inv.y * rt1.z, 0.f);
            float4 m2 = make_float4( 
                scale_inv.z * rt2.x, scale_inv.z * rt2.y, scale_inv.z * rt2.z, 0.f);

            float3 pos_local = data->geo.threedgs.pt_positions[particle_index];
            //todo 还没思路怎么把高斯instance应用localtoworld的transform
            //思路1：把transfrom矩阵看成一个整体推算
            //思路2：在Geometry::Threedgs里面把SRT矩阵也加进去
            float3 pos = pos_local;

            float3 o_g = optix_impl::optixTransformVector(m0, m1, m2, ray_origin - pos);
            float3 d_g = optix_impl::optixTransformVector(m0, m1, m2, ray_direction);
            float  t_max_res = (-dot(o_g, d_g)) / dot(d_g, d_g);
            float3 pt  = ray_origin + ray_direction * t_max_res;

            //R^T * S-1 * S-1
            float3 t0 = make_float3(rt0.x * scale_inv.x * scale_inv.x, 
                rt0.x * scale_inv.y * scale_inv.y, 
                rt0.x * scale_inv.z * scale_inv.z);
            float3 t1 = make_float3(rt1.x * scale_inv.x * scale_inv.x, 
                rt1.x * scale_inv.y * scale_inv.y, 
                rt1.x * scale_inv.z * scale_inv.z);
            float3 t2 = make_float3(rt2.x * scale_inv.x * scale_inv.x, 
                rt2.x * scale_inv.y * scale_inv.y, 
                rt2.x * scale_inv.z * scale_inv.z);

            // * R
            m0 = make_float4(
                t0.x * r0.x + t0.y * r1.x + t0.z * r2.x,
                t0.x * r0.y + t0.y * r1.y + t0.z * r2.y,
                t0.x * r0.z + t0.y * r1.z + t0.z * r2.z,
                0.f);
            m1 = make_float4(
                t1.x * r0.x + t1.y * r1.x + t1.z * r2.x,
                t1.x * r0.y + t1.y * r1.y + t1.z * r2.y,
                t1.x * r0.z + t1.y * r1.z + t1.z * r2.z,
                0.f);
            m2 = make_float4(
                t2.x * r0.x + t2.y * r1.x + t2.z * r2.x,
                t2.x * r0.y + t2.y * r1.y + t2.z * r2.y,
                t2.x * r0.z + t2.y * r1.z + t2.z * r2.z,
                0.f);

            m0 = make_float4(scale_inv.x * scale_inv.x, 0, 0, 0);
            m1 = make_float4(0, scale_inv.y * scale_inv.y, 0, 0);
            m2 = make_float4(0, 0, scale_inv.z * scale_inv.z, 0);
            
            float3 x_minus_mu = pt - pos;
            float3 temp = optix_impl::optixTransformVector(m0, m1, m2, x_minus_mu);
            float  power = -dot(x_minus_mu, temp);
            float alpha_hit = min(0.99f, 
                expf(power) * data->geo.threedgs.opacities[particle_index]);

            if (alpha_hit > PLY_3DGS_ALPHA_MIN) {
                unsigned int index_begin = particle_index * PLY_3DGS_NUM_SHS;
                float3 radiance_hit = ComputeSH(data, index_begin, 
                    ray_direction.x, ray_direction.y, ray_direction.z);
                radiance += transmittance * alpha_hit * radiance_hit;
                transmittance *= (1 - alpha_hit);
            }
            //测试每个包围体第一个顶点位置
            //uint3 idx = data->geo.threedgs.indices[particle_index*20];
            //pos = data->geo.threedgs.positions[idx.x];
            
            //测试中心点是否在包围体中心
            //float t_close = dot(ray_direction, pos - ray_origin) / dot(ray_direction, ray_direction);
            //float3 p_close  = ray_origin + ray_direction * t_close;
            //float  dis      = sqrt(dot(pos - p_close, pos - p_close)) * 10.f;

            //radiance += max(1.f - dis, 0.f);
            //break;
        }
        // visualize test
        //optix_launch_params.albedo_buffer[pixel_index] = 
        //    make_float3((float)hit_count/PLY_3DGS_CHUNK_SIZE);
        ++ray_count;

        // if not enough, it's tmax. if the queue is full, use the furthest one
        t_curr = record.queue.hit_t[PLY_3DGS_CHUNK_SIZE-1];
        //break;
    }
    //if (t_curr == t_max) radiance = make_float3(1, 0, 0);
    if (t_curr != t_max)
        optix_launch_params.normal_buffer[pixel_index] = make_float3(min(t_curr/10.f, 1.f));

    optix_launch_params.albedo_buffer[pixel_index] = make_float3(min((float)ray_count/100, 1.f));

    optix_launch_params.frame_buffer[pixel_index] = make_float4(radiance, 1.f);
}

extern "C" __global__ void __miss__default() {
    auto record = optix::GetPRD<PathPayloadRecord>();
    if (optix_launch_params.emitters.env) {
        auto &env = *optix_launch_params.emitters.env.GetDataPtr();

        const auto ray_dir = normalize(optixGetWorldRayDirection());
        const auto ray_o = optixGetWorldRayOrigin();

        optix::LocalGeometry env_local;
        env_local.position = ray_o + ray_dir;
        optix::EmitEvalRecord emit_record;
        env.Eval(emit_record, env_local, ray_o);
        record->env_radiance = emit_record.radiance;
        record->env_pdf = emit_record.pdf;
    } else {
        record->env_radiance = make_float3(0.f);
    }
    record->done = true;
}
extern "C" __global__ void __miss__shadow() {
    // optixSetPayload_0(0u);
}

__device__ __forceinline__ void ClosestHit() {
    const pt::HitGroupData *sbt_data = (pt::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();
    const auto ray_o = optixGetWorldRayOrigin();

    sbt_data->geo.GetHitLocalGeometry(record->hit.geo, ray_dir, sbt_data->mat.twosided);
    if (sbt_data->emitter_index_offset >= 0) {
        record->hit.emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
    } else {
        record->hit.emitter_index = -1;
    }
    record->hit.bsdf = sbt_data->mat.GetLocalBsdf(record->hit.geo.texcoord);

    record->hit.geo.texcoord = make_float2(optixGetTriangleBarycentrics().x,
                                           optixGetTriangleBarycentrics().y);
}

__device__ __forceinline__ void ClosestHitShadow() {
    optixSetPayload_0(1u);
}

extern "C" __global__ void __closesthit__default() { ClosestHit(); }
extern "C" __global__ void __closesthit__default_sphere() { ClosestHit(); }
extern "C" __global__ void __closesthit__default_curve() { ClosestHit(); }

extern "C" __global__ void __closesthit__shadow() { ClosestHitShadow(); }
extern "C" __global__ void __closesthit__shadow_sphere() { ClosestHitShadow(); }
extern "C" __global__ void __closesthit__shadow_curve() { ClosestHitShadow(); }

extern "C" __global__ void __anyhit__3dgs() { 
    pt::HitGroupData *sbt_data = (pt::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();
    if (sbt_data->geo.Is3dgs()) {
        const auto bounding_face_index = optixGetPrimitiveIndex();
        unsigned int particle_index = bounding_face_index / 20;
        
        unsigned int n_hit = record->queue.hit_count;
        float  t_hit = optixGetRayTmax();
        void* sbt_data_swap = sbt_data;

        // swap insert
        for (auto i = 0u; i < PLY_3DGS_CHUNK_SIZE; ++i) {
            if (t_hit < record->queue.hit_t[i]) {
                unsigned int index_temp    = record->queue.hit_index[i];
                float        t_temp        = record->queue.hit_t[i];
                void*        sbt_temp      = record->queue.sbts[i];
                record->queue.hit_index[i] = particle_index;
                record->queue.hit_t[i]     = t_hit;
                record->queue.sbts[i]      = sbt_data_swap;
                particle_index             = index_temp;
                t_hit                      = t_temp;
                sbt_data_swap              = sbt_temp;
            }
        }
        record->queue.hit_count = n_hit + 1;
        if (optixGetRayTmax() < record->queue.hit_t[PLY_3DGS_CHUNK_SIZE-1])
            optixIgnoreIntersection();
    }
    //else todo 应该是要把射线区间缩小到其他mesh前
    //record->hit.geo.texcoord = make_float2(0.f, 1.f);
    //optixIgnoreIntersection();
}
