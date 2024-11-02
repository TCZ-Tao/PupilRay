#include <optix.h>

#include "optix/util.h"
#include "render/material/bsdf/bsdf.h"

#include "cuda/random.h"
#include "cuda/matrix.h"

#include "resource/3dgs/ply_loader.h"

#include "type.h"

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
    float t_hit;
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
    bool reach_bg;

    float test;
};

CUDA_INLINE CUDA_DEVICE float3 GetSHCoef(pt::HitGroupData* data,
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
CUDA_INLINE CUDA_DEVICE float3 ComputeSH(pt::HitGroupData* data,
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

// quaternion vector to rotation matrix
CUDA_INLINE CUDA_DEVICE void quat2Rot(const float4 q, mat3x3& R) {
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    R.r0 = make_float3(1.f - 2.f * (y * y - z * z), 2.f * (x * y - z * r), 2.f * (x * z + y * r));
    R.r1 = make_float3(2.f * (x * y + z * r), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - x * r));
    R.r2 = make_float3(2.f * (x * z - y * r), 2.f * (y * z + x * r), 1.f - 2.f * (x * x + y * y));
}


CUDA_INLINE CUDA_DEVICE void computeCov2D(const float3 scale, const float4 rot){

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

    record.reach_bg = false;
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
    
    float t_min = 0.001f;
    float t_max = 1e16f; //t_scene_max?


    // for test
            //if (record.done) {
            //    optix_launch_params.frame_buffer[pixel_index] = make_float4(record.env_radiance);
            //} else {
            //    optix_launch_params.frame_buffer[pixel_index] = 
            //        make_float4(record.hit.geo.texcoord.x, record.hit.geo.texcoord.y, 
            //            1.f - record.hit.geo.texcoord.x - record.hit.geo.texcoord.y, 1.f);
            //}
    int depth = 0;
    bool reach_bg = false;
    float3 ray_origin_next = ray_origin;
    float3 ray_direction_next = ray_direction;
    optix::BsdfSamplingRecord bsdf_sample_record_pre; // bsdf at ray begin
    bool direct_light = false;

    while (!reach_bg) {
        // ===== trace common objects
        optixTrace(optix_launch_params.handle,
                    ray_origin, ray_direction,
                    t_min, t_max, 0.f,
                    OptixVisibilityMask(optix::VISIBILITY_MASK_COMMON), 
                    OPTIX_RAY_FLAG_NONE,
                    0, 2, 0,
                    u0, u1);
        auto local_hit = record.hit;
        auto throughput_old = record.throughput;
        float3 direct_emission;

        optix::BsdfSamplingRecord bsdf_sample_record;
        if (!record.reach_bg) {
            // ===== hit emitter
            if (local_hit.emitter_index >= 0 && !direct_light) {
                auto& emitter = optix_launch_params.emitters.areas[local_hit.emitter_index];
                direct_emission = emitter.GetRadiance(local_hit.geo.texcoord);
                direct_light = true;
                //break;
            }

            // ===== direct light sampling
            auto& emitter = optix_launch_params.emitters.SelectOneEmiiter(record.random.Next());
            optix::EmitterSampleRecord emitter_sample_record;
            emitter.SampleDirect(emitter_sample_record, local_hit.geo, record.random.Next2());

            bool occluded = optix::Emitter::TraceShadowRay(
                optix_launch_params.handle,
                local_hit.geo.position,
                emitter_sample_record.wi,
                0.0001f,
                emitter_sample_record.distance - 0.0001f
            );
            if (!occluded) {
                optix::BsdfSamplingRecord eval_record;
                eval_record.wi = optix::ToLocal(emitter_sample_record.wi, local_hit.geo.normal);
                eval_record.wo = optix::ToLocal(-ray_direction, local_hit.geo.normal);
                eval_record.sampler = &record.random;
                record.hit.bsdf.Eval(eval_record);
                float3 f = eval_record.f;
                float  pdf = eval_record.pdf;
                if (!optix::IsZero(f * emitter_sample_record.pdf)) {
                    float NoL = dot(local_hit.geo.normal, emitter_sample_record.wi);
                    if (NoL > 0.f) {
                        float mis = emitter_sample_record.is_delta ? 
                            1.f : optix::MISWeight(emitter_sample_record.pdf, pdf);
                        emitter_sample_record.pdf *= emitter.select_probability;
                        record.radiance += record.throughput * emitter_sample_record.radiance 
                            * f * NoL * mis / emitter_sample_record.pdf;
                        // 目前高斯不影响光源的光照
                    }
                }
            }

            // ===== bsdf sampling
            bsdf_sample_record.wo = optix::ToLocal(-ray_direction, local_hit.geo.normal);
            bsdf_sample_record.sampler = &record.random;
            record.hit.bsdf.Sample(bsdf_sample_record);

            if ( !direct_light &&
                (optix::IsZero(bsdf_sample_record.f * abs(bsdf_sample_record.wi.z)) 
                || optix::IsZero(bsdf_sample_record.pdf)))
                break;
            
            record.throughput *= 
                bsdf_sample_record.f * abs(bsdf_sample_record.wi.z) / bsdf_sample_record.pdf;

            // next ray
            ray_origin_next = record.hit.geo.position;
            ray_direction_next = optix::ToWorld(bsdf_sample_record.wi, local_hit.geo.normal);
        }
        // avoid being changed in gaussian tracing
        auto env_radiance = record.env_radiance;
        auto env_pdf      = record.env_pdf;
        reach_bg = record.reach_bg;

        // ===== trace gaussians
        float3 radiance_gaussian = make_float3(0.f);
        float  transmittance = 1.f;
        float  transmittance_min = 0.01f;
        {
            float  t_curr = t_min;
            float  t_curr_max = record.hit.t_hit;

            auto ray_count = 0u;

            record.reach_bg = false;

            while (t_curr < t_curr_max && !record.reach_bg && transmittance > transmittance_min) {
                // reset kbuffer
                record.queue.hit_count = 0;
                for (int i = 0; i < PLY_3DGS_CHUNK_SIZE; ++i)
                    record.queue.hit_t[i] = t_curr_max;

                optixTrace(optix_launch_params.handle,
                    ray_origin, ray_direction,
                    t_curr, t_curr_max, 0.f,
                    OptixVisibilityMask(optix::VISIBILITY_MASK_GAUSSIAN), 
                    OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                    0, 2, 0,
                    u0, u1);

                unsigned int hit_count = record.queue.hit_count;
                const unsigned int queue_size = min(PLY_3DGS_CHUNK_SIZE, hit_count);
                for (int i = 0; i < queue_size; ++i) {
                    const unsigned int particle_index = record.queue.hit_index[i];
                    pt::HitGroupData* data = static_cast<pt::HitGroupData*>(record.queue.sbts[i]);

                    // compute kernal response

                    float3 pos_local = data->geo.threedgs.pt_positions[particle_index];
                    //todo 还没思路怎么把高斯instance应用localtoworld的transform
                    //思路1：把transfrom矩阵看成一个整体推算
                    //思路2：在Geometry::Threedgs里面把SRT矩阵也加进去
                    //要不先只加平移
                    float3 pos = pos_local;

                    float3 scale = data->geo.threedgs.scales[particle_index] *
                        optix_launch_params.config.scale_factor;
                    float3 scale_inv = 1.f / scale;
                    mat3x3 Sinv = make_mat3x3(scale_inv);

                    mat3x3 R;
                    quat2Rot(data->geo.threedgs.rotations[particle_index], R);
                    mat3x3 RT = transpose(R);

                    // Covariance
                    mat3x3 RSinv2; // R * S-1*S-1
                    RSinv2.r0 = R.r0 * scale_inv * scale_inv;
                    RSinv2.r1 = R.r1 * scale_inv * scale_inv;
                    RSinv2.r2 = R.r2 * scale_inv * scale_inv;
                    mat3x3 Sigma  = RSinv2 * RT;

                    // Max response from 3DGRT
                    mat3x3 SinvRT = Sinv * RT; // S^{-1}*R^T
                    float3 o_g = SinvRT * (ray_origin - pos);
                    float3 d_g = SinvRT * ray_direction;
                    float t_max_res = (-dot(o_g, d_g)) / dot(d_g, d_g);
                    float3 pt = ray_origin + ray_direction * t_max_res;
    
                    float3 x_minus_mu = pt - pos;
                    float3 temp = Sigma * x_minus_mu;
                    float  power = -dot(x_minus_mu, temp);

                    //todo 原版3DGS投影方式
                    //float4 p_hom = optix_launch_params.projection_matrix * make_float4(pos, 1.f);
                    //float  p_w   = 1.f / (p_hom.w + 0.0000001f);
                    //float3 p_proj = make_float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);
                    // computeCov2D


                    float alpha_hit = min(0.99f, 
                        expf(power) * data->geo.threedgs.opacities[particle_index]);

                    if (alpha_hit > PLY_3DGS_ALPHA_MIN) {
                        unsigned int index_begin = particle_index * PLY_3DGS_NUM_SHS;
                        float3 radiance_hit = ComputeSH(data, index_begin, 
                            ray_direction.x, ray_direction.y, ray_direction.z);
                        radiance_gaussian += transmittance * alpha_hit * radiance_hit;
                        transmittance *= (1 - alpha_hit);
                    }
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
            } //while
        }

        if (transmittance == transmittance_min) transmittance = 0.f;

        if (direct_light) {
            record.radiance += direct_emission * transmittance;
            break;
        }

        if (!optix::IsZero(radiance_gaussian))
            record.radiance += throughput_old * radiance_gaussian; 
            // lerp by transmittance?

        // environment light
        if (reach_bg) {
            float mis = optix::MISWeight(bsdf_sample_record_pre.pdf, env_pdf);
            record.env_radiance = env_radiance * 
                (optix::IsZero(bsdf_sample_record_pre.pdf) ? 
                    make_float3(1.f) : (record.throughput * mis))
                * transmittance;
            break;
        }

        // hit an emitter object
        if (local_hit.emitter_index >= 0) {
            auto& emitter = optix_launch_params.emitters.areas[local_hit.emitter_index];
            optix::EmitEvalRecord emit_record;
            emitter.Eval(emit_record, local_hit.geo, ray_origin);
            if (!optix::IsZero(emit_record.pdf)) {
                float mis = bsdf_sample_record_pre.sampled_type & optix::EBsdfLobeType::Delta ?
                    1.f : 
                    optix::MISWeight(bsdf_sample_record_pre.pdf, 
                        emit_record.pdf * emitter.select_probability);
                record.radiance += record.throughput * emit_record.radiance * mis
                    * transmittance;
            }
        }

        // terminate
        ++depth;
        if (depth >= optix_launch_params.config.max_depth) break;

        float rr = depth > 2 ? 0.95f : 1.f;
        record.throughput /= rr;
        if (record.random.Next() > rr) break;

        // next
        ray_origin = ray_origin_next;
        ray_direction = ray_direction_next;
        bsdf_sample_record_pre = bsdf_sample_record;

        // reset
        record.reach_bg = false;
    }

    record.radiance += record.env_radiance;
    // accumulate multiple samples
    if (optix_launch_params.config.accumulated_flag && optix_launch_params.sample_cnt > 0) {
        const float t = 1.f / (optix_launch_params.sample_cnt + 1.f);
        const float3 pre = make_float3(optix_launch_params.accum_buffer[pixel_index]);
        record.radiance  = lerp(pre, record.radiance, t);
    }
    optix_launch_params.accum_buffer[pixel_index] = make_float4(record.radiance, 1.f);
    optix_launch_params.frame_buffer[pixel_index] = make_float4(record.radiance, 1.f);

    /*
    //if (t_curr == t_max) radiance = make_float3(1, 0, 0);
    if (t_curr != t_max)
        optix_launch_params.normal_buffer[pixel_index] = make_float3(min(t_curr/10.f, 1.f));

    optix_launch_params.albedo_buffer[pixel_index] = make_float3(min((float)ray_count/100, 1.f));

    optix_launch_params.frame_buffer[pixel_index] = make_float4(radiance, 1.f);
    */
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
        // 不应该到这里，场景里至少写一个ConstantEnvEmitter
    }
    record->hit.t_hit = optixGetRayTmax();
    record->reach_bg = true;
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
    record->hit.t_hit = optixGetRayTmax();

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
