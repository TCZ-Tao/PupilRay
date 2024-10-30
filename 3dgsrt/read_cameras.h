#pragma once
#include <vector>
#include <filesystem>
#include "glm/glm.hpp"

struct CameraInfo {
    int                        uid;
    glm::mat3                  R;
    glm::vec3                  T;
    float                      FovY;
    float                      FovX;
    std::vector<unsigned char> image;
    std::string                image_path;
    std::string                image_name;
    int                        width;
    int                        height;

    static float fov2focal(float fov, float pixels);
    static float focal2fov(float focal, float pixels); 
};

struct BasicPointCloud {
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> colors;
    std::vector<glm::vec3> normals;
};

struct SceneInfo {
    BasicPointCloud         point_cloud;
    std::vector<CameraInfo> train_cameras;
    std::vector<CameraInfo> test_cameras;
    // nerf_normalization
    glm::vec3 translate;
    float     radius;

    std::string ply_path;
};


glm::mat4 getWorld2View2(const glm::mat3& R, const glm::vec3& t, 
    const glm::vec3& translate = glm::vec3(0.0f), float scale = 1.0f);

// store camera, image, transform. for using.
class CamWithImg {
public:
    int uid;
    int colmap_id;
    glm::mat3 R;
    glm::vec3 T;
    float FoVx;
    float FoVy;
    std::string image_name;
    std::vector<unsigned char> original_image;
    int image_width;
    int image_height;
    glm::vec3 trans;
    float scale;
    glm::mat4 world_view_transform;
    glm::vec3 camera_center;

    CamWithImg(int colmap_id, 
        const glm::mat3& R, const glm::vec3& T, 
        float FoVx, float FoVy, 
        const std::vector<unsigned char>& image, const std::string& image_name, 
        float width, float height, int uid, 
        const glm::vec3& trans = glm::vec3(0.0, 0.0, 0.0), float scale = 1.0) 
        : uid(uid), colmap_id(colmap_id), R(R), T(T), FoVx(FoVx), FoVy(FoVy), 
        image_name(image_name), original_image(image), 
        image_width(width), image_height(height),
         trans(trans), scale(scale) {
        world_view_transform = glm::transpose(getWorld2View2(R, T, trans, scale));
        camera_center = glm::vec3(glm::inverse(world_view_transform)[3]);
    }
};

class ReadCameras {
    void getNerfppNorm(const std::vector<CameraInfo>& cam_info,
                       glm::vec3&                     translate,
                       float&                         radius);

    std::vector<CameraInfo> readCamerasFromTransforms(const std::string& path,
                                                      const std::string& transformsfile,
                                                      bool               white_background,
                                                      const std::string& extension = ".png");

    BasicPointCloud fetchPly(const std::string& path);

    SceneInfo readNerfSyntheticInfo(const std::string& path, bool white_background, 
        bool eval, const std::string& extension = ".png");

    std::vector<CamWithImg> cameraList_from_camInfos(const std::vector<CameraInfo>& cam_infos);

public:
    ReadCameras(const std::filesystem::path path, bool white_background, 
        bool eval, const std::string& extension = ".png") ;

    std::vector<CameraInfo> cam_info_list;
    std::vector<CamWithImg> train_cameras;
    std::vector<CamWithImg> test_cameras;
};
