#include "read_cameras.h"

#include <fstream>

#include "json.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "util/log.h"

using json = nlohmann::json;


float CameraInfo::fov2focal(float fov, float pixels) {
    return pixels / (2.f * glm::tan(fov / 2.f));
}
float CameraInfo::focal2fov(float focal, float pixels) {
    return 2 * glm::atan(pixels / (2.f * focal));
}

glm::mat4 getWorld2View2(const glm::mat3& R, const glm::vec3& t, 
    const glm::vec3& translate, float scale) {
    glm::mat4 Rt = glm::transpose(glm::mat4(glm::mat3(R)));
    Rt[3] = glm::vec4(t, 1.0f);

    glm::mat4 C2W = glm::inverse(Rt);
    glm::vec3 cam_center = glm::vec3(C2W[3]);
    cam_center = (cam_center + translate) * scale;
    C2W[3] = glm::vec4(cam_center, 1.0f);
    Rt = glm::inverse(C2W);
    return Rt;
}


void ReadCameras::getNerfppNorm(const std::vector<CameraInfo>& cam_info,
                   glm::vec3& translate, float& radius) {
    auto get_center_and_diag = [](const std::vector<glm::vec3>& cam_centers) {
        glm::vec3 center(0.0f);
        for (const auto& cam_center : cam_centers) {
            center += cam_center;
        }
        center /= static_cast<float>(cam_centers.size());

        float diagonal = 0.0f;
        for (const auto& cam_center : cam_centers) {
            float dist = glm::distance(cam_center, center);
            if (dist > diagonal) {
                diagonal = dist;
            }
        }
        return std::make_pair(center, diagonal);
    };

    std::vector<glm::vec3> cam_centers;
    for (const auto& cam : cam_info) {
        glm::mat4 W2C = getWorld2View2(cam.R, cam.T);
        glm::mat4 C2W = glm::inverse(W2C);
        cam_centers.push_back(glm::vec3(C2W[3]));
    }

    auto [center, diagonal] = get_center_and_diag(cam_centers);
    radius = diagonal * 1.1f;
    translate = -center;
}

std::vector<CameraInfo> ReadCameras::readCamerasFromTransforms(const std::string& path, 
    const std::string& transformsfile, bool white_background, 
    const std::string& extension) {
    std::vector<CameraInfo> cam_infos;

    std::ifstream json_file(path + "/" + transformsfile);
    json contents = json::parse(json_file);

    float fovx = contents["camera_angle_x"];
    auto  frames = contents["frames"];

    for (int idx = 0; idx < frames.size(); ++idx) {
        std::string cam_name = path + "/" + frames[idx]["file_path"].get<std::string>() + extension;

        glm::mat4 c2w = glm::make_mat4(frames[idx]["transform_matrix"].get<std::vector<float>>().data());
        c2w[1][1] *= -1;
        c2w[1][2] *= -1;
        c2w[2][1] *= -1;
        c2w[2][2] *= -1;

        glm::mat4 w2c = glm::inverse(c2w);
        glm::mat3 R = glm::transpose(glm::mat3(w2c));
        glm::vec3 T = glm::vec3(w2c[3]);

        std::string image_path = path + "/" + cam_name;
        std::string image_name = std::filesystem::path(cam_name).stem().string();

        int width, height, channels;
        unsigned char* image_data = stbi_load(image_path.c_str(), &width, &height, &channels, 4);

        glm::vec3 bg = white_background ? glm::vec3(1.0f) : glm::vec3(0.0f);
        glm::vec3* norm_data = new glm::vec3[width * height];

        for (int i = 0; i < width * height; ++i) {
            norm_data[i] = glm::vec3(image_data[i * 4] / 255.0f, 
                image_data[i * 4 + 1] / 255.0f, 
                image_data[i * 4 + 2] / 255.0f);
            float alpha = image_data[i * 4 + 3] / 255.0f;
            norm_data[i] = norm_data[i] * alpha + bg * (1 - alpha);
        }

        std::vector<unsigned char> image(width * height * 3);
        for (int i = 0; i < width * height; ++i) {
            image[i * 3] = static_cast<unsigned char>(norm_data[i].r * 255.0f);
            image[i * 3 + 1] = static_cast<unsigned char>(norm_data[i].g * 255.0f);
            image[i * 3 + 2] = static_cast<unsigned char>(norm_data[i].b * 255.0f);
        }

        float fovy = CameraInfo::focal2fov(CameraInfo::fov2focal(fovx, width), height);

        cam_infos.push_back({idx, R, T, fovy, fovy, image, image_path, image_name, width, height});

        stbi_image_free(image_data);
        delete[] norm_data;
    }

    return cam_infos;
}

BasicPointCloud ReadCameras::fetchPly(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, 
        aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
    if (scene == nullptr) Pupil::Log::Error("no initial point cloud file!");
    
    BasicPointCloud pointCloud;

    if (scene && scene->HasMeshes()) {
        for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[i];
            for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
                aiVector3D pos = mesh->mVertices[j];
                pointCloud.points.emplace_back(pos.x, pos.y, pos.z);

                if (mesh->HasVertexColors(0)) {
                    aiColor4D color = mesh->mColors[0][j];
                    pointCloud.colors.emplace_back(color.r, color.g, color.b);
                } else { // Default to white if no color
                    pointCloud.colors.emplace_back(1.0f, 1.0f, 1.0f); 
                }

                if (mesh->HasNormals()) {
                    aiVector3D normal = mesh->mNormals[j];
                    pointCloud.normals.emplace_back(normal.x, normal.y, normal.z);
                } else {
                    pointCloud.normals.emplace_back(0.0f, 0.0f, 0.0f); // Default normal
                }
            }
        }
    }

    return pointCloud;
}

SceneInfo ReadCameras::readNerfSyntheticInfo(const std::string& path, bool white_background, 
    bool eval, const std::string& extension) {
    Pupil::Log::Info("Reading Training Transforms");
    auto train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", 
        white_background, extension);
    Pupil::Log::Info("Reading Test Transforms");
    auto test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", 
        white_background, extension);

    if (!eval) {
        train_cam_infos.insert(train_cam_infos.end(), test_cam_infos.begin(), test_cam_infos.end());
        test_cam_infos.clear();
    }

    glm::vec3 translate;
    float     radius;
    getNerfppNorm(train_cam_infos, translate, radius);

    // 读取初始点云
    //todo 点云好像只用来创建初始高斯云，确认后去掉这部分，用python脚本来创建
    std::string ply_path = path + "/points3d.ply";
    BasicPointCloud pcd = fetchPly(ply_path);

    return SceneInfo { pcd, train_cam_infos, test_cam_infos, translate, radius, ply_path };
}

ReadCameras::ReadCameras(const std::filesystem::path path, bool white_background, 
    bool eval, const std::string& extension) {
    SceneInfo scene_info = readNerfSyntheticInfo(path.string(), false, false, ".png");

    // 这两个似乎没用
    if (!scene_info.test_cameras.empty()) 
        cam_info_list.insert(cam_info_list.end(), scene_info.test_cameras.begin(), 
            scene_info.test_cameras.end());
    
    if (!scene_info.train_cameras.empty()) 
        cam_info_list.insert(cam_info_list.end(), scene_info.train_cameras.begin(), 
            scene_info.train_cameras.end());
    
    //float cameras_extent = scene_info.radius;

    train_cameras = cameraList_from_camInfos(scene_info.train_cameras);
    test_cameras = cameraList_from_camInfos(scene_info.test_cameras);
}

std::vector<CamWithImg> ReadCameras::cameraList_from_camInfos(
    const std::vector<CameraInfo>& cam_infos) {
    std::vector<CamWithImg> camera_list;

    for (int id = 0; id < cam_infos.size(); ++id) {
        auto cam_info = cam_infos[id];
        camera_list.push_back(CamWithImg(cam_info.uid, cam_info.R, cam_info.T, 
            cam_info.FovX, cam_info.FovY, 
            cam_info.image, cam_info.image_name, 
            cam_info.width, cam_info.height, id, 
            glm::vec3(0.0, 0.0, 0.0), 1.0));
    }

    return camera_list;
}
