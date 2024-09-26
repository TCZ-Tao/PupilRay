#include "system/system.h"
#include "pt_pass.h"
#include "static.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        auto pt_pass = std::make_unique<Pupil::pt::PTPass>();
        system->AddPass(pt_pass.get());

        std::filesystem::path tdgs_file_path {
            "D:/Source/gaussian-splatting/output/102d4d42-d/point_cloud/iteration_7000/point_cloud.ply"
        };
        std::filesystem::path scene_file_path{"D:/Source/PupilRay/test_data"};
        scene_file_path /= "test.xml";

        system->SetScene(scene_file_path);

        system->Run();
    }

    system->Destroy();

    return 0;
}