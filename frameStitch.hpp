//
//  frameStitch.hpp
//  opencv
//
//  Created by yu zhang on 16/6/13.
//  Copyright © 2016年 yu zhang. All rights reserved.
//

#ifndef frameStitch_hpp
#define frameStitch_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <opencv2/calib3d/calib3d.hpp>

#include "bundle.hpp"
#include "FileStorageExt.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

#define G_INITIAL 0

struct CV_EXPORTS ImageKeypoint
{
    int img_idx;
    KeyPoint keypoint;
    Point2f frameUV;
};

class frameStitch {
public:
    frameStitch();
    
    ~frameStitch();
    
    void getImgNames(vector<String>);
    
    bool resizeImages();
    
    void findFeatures();
    
    void pairwiseMatch();
    
    void waveCorrection(int); //write
    
    void waveCorrectionFrame();
    
    bool computeCameras(int);
    
    void readCameras(int); //read bundled cameara params
    
    void readCamerasOri(int); //read ori camera params
    
    bool warpImages();
    
    void compensateExposure();
    
    void findSeamMask();
    
    void blendImages(int, bool, string);
    
    // frame related function
    
    void findFrameFeature(string);
    
    void pairwiseFrameMatch(string);
    
    bool computeFrameCamera();
    
    bool computeFrameBundleAdjustment();
    
    void computeFrameCameraPos();
    
    void computeFrameCameraPosV2();
    
    void getPairKeypoints();
        
protected:
    vector<String> img_names;
    bool preview = false;
    bool try_cuda = false;
    double work_megapix = 0.6;
    double seam_megapix = 0.1;
    double compose_megapix = -1;
    float conf_thresh = 1.f;
    string features_type = "surf";
    string ba_cost_func = "reproj";
    string ba_refine_mask = "xxxxx";
    bool do_wave_correct = true;
    WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
    bool save_graph = false;
    std::string save_graph_to;
    string warp_type = "spherical";
    int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
    float match_conf = 0.3f;
    string seam_find_type = "gc_color";
    int blend_type = Blender::NO;
    int timelapse_type = Timelapser::AS_IS;
    float blend_strength = 5;
    string result_name;
    bool timelapse = false;
    int range_width = -1;
    
    double work_scale = 1;
    double seam_scale = 1;
    double compose_scale = 1;
    bool is_work_scale_set = false;
    bool is_seam_scale_set = false;
    bool is_compose_scale_set = false;
    double seam_work_aspect = 1;
    
    float warped_image_scale;
    
    vector<ImageFeatures> features;
    vector<Mat> images;
    vector<Mat> full_images;
    vector<Size> full_img_sizes;
    vector<MatchesInfo> pairwise_matches;
    vector<CameraParams> cameras;
    vector<int> indices;
    
    vector<Point> corners;
    vector<UMat> masks_warped;
    vector<UMat> images_warped;
    vector<Size> sizes;

    int num_images;
    
    Ptr<RotationWarper> warper;
    Ptr<WarperCreator> warper_creator;
    Ptr<ExposureCompensator> compensator;
    
    ImageFeatures frameFeature;
    vector<MatchesInfo> pairwise_frame_matches;
    vector<MatchesInfo> pairwise_frame_matches_three;
    vector<ImageFeatures> allFeatures;
    vector<ImageFeatures> threeFeatures;
    vector<ImageKeypoint> g_imgKeypoint;
};


#endif /* frameStitch_hpp */
