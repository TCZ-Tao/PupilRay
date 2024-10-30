#include "system/system.h"
#include "static.h"
#include "pt_pass.h"
#include "resource/3dgs/ply_loader.h"

#include "read_cameras.h"
#include <cstdlib>// for rand()

void train(std::filesystem::path train_file_path) {
    // read cameras
    ReadCameras readCameras(train_file_path.string(), false, false, ".png");
    std::vector<CamWithImg> viewpoint_stack = readCameras.train_cameras;

    // read initial particles
    auto gaussian_ply_path = train_file_path / "init.ply";
    Pupil::resource::PlyLoader ply_loader;
    //ply_loader.LoadFromFileForTraining(gaussian_ply_path.string());

    const auto iteration_num = 30000u;
    for (auto iter = 0u; iter < iteration_num; ++iter) {
        // update learning rate

        // sample a viewpoint
        int random_index = rand() % viewpoint_stack.size();
        CamWithImg viewpoint_cam   = viewpoint_stack[random_index];
        viewpoint_stack.erase(viewpoint_stack.begin() + random_index);

        // render
        //只渲染一张

        // calculate image loss

        // backpropagation
    }
}

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        auto pt_pass = std::make_unique<Pupil::pt::PTPass>();
        system->AddPass(pt_pass.get());
        std::filesystem::path scene_file_path{Pupil::DATA_DIR};
        scene_file_path /= "test_data/test.xml";

        //Pupil::Log::Info("test");

        system->SetScene(scene_file_path);

        system->Run();
    }

    system->Destroy();

    return 0;
}
