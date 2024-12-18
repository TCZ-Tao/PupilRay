#include "pt_pass.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/system.h"
#include "system/gui/gui.h"
#include "world/world.h"
#include "world/render_object.h"

extern "C" char embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
    int m_max_depth;
    float m_scale_factor;
    bool m_accumulated_flag;

Pupil::world::World *m_world = nullptr;
}// namespace

namespace Pupil::pt {
PTPass::PTPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void PTPass::OnRun() noexcept {
    if (m_dirty) {
        m_optix_launch_params.camera.SetData(m_world_camera->GetCudaMemory());
        // 因为获取不到camera_dirty状态，暂时每帧都更新一次
        auto w2c_temp = m_world_camera->GetViewMatrix();
        auto proj_temp = m_world_camera->GetProjectionMatrix();
        //proj_temp.r1 *= -1.f;
        auto mat_temp = m_world_camera->GetProjectionMatrix() * m_world_camera->GetViewMatrix();
        //mat_temp.r1 *= -1.f;

        // debug print projection matrix
        //printf("projection matrix:\n");
        //m_world_camera->GetProjectionMatrix().Print();
        //printf("view matrix:\n");
        //m_world_camera->GetViewMatrix().Print();
        //printf("c2w matrix:\n");
        //m_world_camera->GetToWorldMatrix().Print();
        //printf("view proj matrix:\n");
        //mat_temp.Print();

        m_optix_launch_params.view_matrix = ToCudaType(w2c_temp);
        m_optix_launch_params.proj_matrix = ToCudaType(proj_temp);
        m_optix_launch_params.full_proj_matrix = ToCudaType(mat_temp);

        m_optix_launch_params.config.max_depth = m_max_depth;
        m_optix_launch_params.config.scale_factor = m_scale_factor;
        m_optix_launch_params.config.accumulated_flag = m_accumulated_flag;
        m_optix_launch_params.sample_cnt = 0;
        m_optix_launch_params.random_seed = 0;
        m_optix_launch_params.handle = m_world->GetIASHandle(2, true);
        m_optix_launch_params.emitters = m_world->emitters->GetEmitterGroup();
        m_dirty = false;
    }

    m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                      m_optix_launch_params.config.frame.height);
    m_optix_pass->Synchronize();

    m_optix_launch_params.sample_cnt += m_optix_launch_params.config.accumulated_flag;
    ++m_optix_launch_params.random_seed;
}

void PTPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    // auto curve_linear_module = module_mngr->GetModule(optix::EModuleBuiltinType::RoundLinearPrimitive);
    // auto curve_quadratic_module = module_mngr->GetModule(optix::EModuleBuiltinType::RoundQuadraticBsplinePrimitive);
    auto curve_cubic_module = module_mngr->GetModule(optix::EModuleBuiltinType::RoundCubicBsplinePrimitive);
    // auto curve_catmullrom_module = module_mngr->GetModule(optix::EModuleBuiltinType::RoundCatmullromPrimitive);
    auto pt_module = module_mngr->GetModule(embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    { // for mesh(triangle) geo
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .ray_gen_entry = "__raygen__main",
            .miss_entry = "__miss__default",
            .hit_group = { .ch_entry = "__closesthit__default" },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .miss_entry = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__shadow" },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }

    {
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__default_sphere",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__shadow_sphere",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }

    {   // for 3dgs
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ah_entry = "__anyhit__3dgs" },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
    }

    auto curve_module = curve_cubic_module;
    {
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__default_curve",
                           .intersect_module = curve_module },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__shadow_curve",
                           .intersect_module = curve_module },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }

    {
        auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
        pipeline_desc.callable_programs.insert(
            pipeline_desc.callable_programs.end(),
            mat_programs.begin(), mat_programs.end());
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void PTPass::SetScene(world::World *world) noexcept {
    m_world_camera = world->camera.get();

    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;
    m_optix_launch_params.config.max_depth = world->scene->integrator.max_depth;
    m_optix_launch_params.config.scale_factor     = 1.f;
    m_optix_launch_params.config.accumulated_flag = true;

    auto cam_desc = m_world_camera->GetDesc();
    //todo fov更新也加到dirty那里
    m_optix_launch_params.fov.y = cam_desc.fov_y * 3.14159265358979323846f / 180;
    float fov_x = 2.f * atanf(tanf(m_optix_launch_params.fov.y * 0.5f) * cam_desc.aspect_ratio);
    m_optix_launch_params.fov.x = fov_x;
    m_optix_launch_params.fov.z = tanf(fov_x * 0.5f);
    m_optix_launch_params.fov.w = tanf(m_optix_launch_params.fov.y * 0.5f);

    m_max_depth = m_optix_launch_params.config.max_depth;
    m_scale_factor = m_optix_launch_params.config.scale_factor;
    m_accumulated_flag = m_optix_launch_params.config.accumulated_flag;

    m_optix_launch_params.random_seed = 0;
    m_optix_launch_params.sample_cnt = 0;

    m_output_pixel_num = m_optix_launch_params.config.frame.width *
                         m_optix_launch_params.config.frame.height;
    auto buf_mngr = util::Singleton<BufferManager>::instance();
    {
        m_optix_launch_params.frame_buffer.SetData(
            buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME)->cuda_ptr,
            m_output_pixel_num);

        BufferDesc desc{
            .name = "pt accum buffer",
            .flag = EBufferFlag::None,
            .width = static_cast<uint32_t>(world->scene->sensor.film.w),
            .height = static_cast<uint32_t>(world->scene->sensor.film.h),
            .stride_in_byte = sizeof(float) * 4
        };
        m_optix_launch_params.accum_buffer.SetData(
            buf_mngr->AllocBuffer(desc)->cuda_ptr, m_output_pixel_num);

        desc.name = "albedo";
        desc.flag = EBufferFlag::AllowDisplay;
        desc.stride_in_byte = sizeof(float3);
        m_optix_launch_params.albedo_buffer.SetData(
            buf_mngr->AllocBuffer(desc)->cuda_ptr, m_output_pixel_num);

        desc.name = "normal";
        m_optix_launch_params.normal_buffer.SetData(
            buf_mngr->AllocBuffer(desc)->cuda_ptr, m_output_pixel_num);
    }

    m_world = world;
    m_optix_launch_params.handle = m_world->GetIASHandle(2, true);

    {
        optix::SBTDesc<SBTTypes> desc{};
        desc.ray_gen_data = {
            .program = "__raygen__main"
        };
        {
            int emitter_index_offset = 0;
            using HitGroupDataRecord = optix::ProgDataDescPair<SBTTypes::HitGroupDataType>;
            for (auto &&ro : world->GetRenderobjects()) {
                HitGroupDataRecord hit_default_data{};
                if (ro->geo.type == optix::Geometry::EType::TriMesh)
                    hit_default_data.program = "__closesthit__default";
                else if (ro->geo.type == optix::Geometry::EType::ThreeDimGaussian)
                    //hit_default_data.program = "__closesthit__default";
                    hit_default_data.program = "__anyhit__3dgs";
                else if (ro->geo.type == optix::Geometry::EType::Sphere)
                    hit_default_data.program = "__closesthit__default_sphere";
                else
                    hit_default_data.program = "__closesthit__default_curve";
                hit_default_data.data.mat = ro->mat;
                hit_default_data.data.geo = ro->geo;
                if (ro->is_emitter) {
                    hit_default_data.data.emitter_index_offset = emitter_index_offset;
                    emitter_index_offset += ro->sub_emitters_num;
                }

                desc.hit_datas.push_back(hit_default_data);

                HitGroupDataRecord hit_shadow_data{};
                if (ro->geo.type == optix::Geometry::EType::TriMesh)
                    hit_shadow_data.program = "__closesthit__shadow";
                else if (ro->geo.type == optix::Geometry::EType::Sphere)
                    hit_shadow_data.program = "__closesthit__shadow_sphere";
                else
                    hit_shadow_data.program = "__closesthit__shadow_curve";

                hit_shadow_data.data.mat.type = ro->mat.type;
                desc.hit_datas.push_back(hit_shadow_data);
            }
        }
        {
            optix::ProgDataDescPair<SBTTypes::MissDataType> miss_data = {
                .program = "__miss__default"
            };
            desc.miss_datas.push_back(miss_data);
            optix::ProgDataDescPair<SBTTypes::MissDataType> miss_shadow_data = {
                .program = "__miss__shadow"
            };
            desc.miss_datas.push_back(miss_shadow_data);
        }
        {
            auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
            for (auto &mat_prog : mat_programs) {
                if (mat_prog.cc_entry) {
                    optix::ProgDataDescPair<SBTTypes::CallablesDataType> cc_data = {
                        .program = mat_prog.cc_entry
                    };
                    desc.callables_datas.push_back(cc_data);
                }
                if (mat_prog.dc_entry) {
                    optix::ProgDataDescPair<SBTTypes::CallablesDataType> dc_data = {
                        .program = mat_prog.dc_entry
                    };
                    desc.callables_datas.push_back(dc_data);
                }
            }
        }
        m_optix_pass->InitSBT(desc);
    }

    m_dirty = true;
}

void PTPass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        m_dirty = true;
    });

    EventBinder<EWorldEvent::RenderInstanceUpdate>([this](void *) {
        m_dirty = true;
    });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((world::World *)p);
    });
}

void PTPass::Inspector() noexcept {
    Pass::Inspector();
    ImGui::Text("sample count: %d", m_optix_launch_params.sample_cnt + 1);
    ImGui::InputInt("max trace depth", &m_max_depth);
    m_max_depth = clamp(m_max_depth, 1, 128);
    if (m_optix_launch_params.config.max_depth != m_max_depth) {
        m_dirty = true;
    }

    ImGui::InputFloat("scale factor", &m_scale_factor);
    m_scale_factor = clamp(m_scale_factor, 0.1f, 10.f);
    if (m_optix_launch_params.config.scale_factor - m_scale_factor > 0.0001f) m_dirty = true;

    if (ImGui::Checkbox("accumulate radiance", &m_accumulated_flag)) {
        m_dirty = true;
    }
}
}// namespace Pupil::pt