//
//  frameStitch.cpp
//  opencv
//
//  Created by yu zhang on 16/6/13.
//  Copyright © 2016年 yu zhang. All rights reserved.
//

#include "frameStitch.hpp"

frameStitch::frameStitch(){
    preview = false;
    try_cuda = false;
    work_megapix = 1.0;
    seam_megapix = 0.1;
    compose_megapix = -1;
    conf_thresh = 1.f;
    features_type = "surf";
    ba_cost_func = "reproj";
    do_wave_correct = true;
    wave_correct = detail::WAVE_CORRECT_HORIZ;
    save_graph = false;
    warp_type = "spherical";
    expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
    match_conf = 0.3f;
    seam_find_type = "gc_color";
    blend_type = Blender::NO;
    timelapse_type = Timelapser::AS_IS;
    blend_strength = 5;
    result_name = "result_";
    timelapse = false;
    range_width = -1;
    
    work_scale = 1;
    seam_scale = 1;
    compose_scale = 1;
    is_work_scale_set = false;
    is_seam_scale_set = false;
    is_compose_scale_set = false;
    seam_work_aspect = 1;
    
}

frameStitch::~frameStitch(){
    img_names.clear();
    features.clear();
    images.clear();
    full_images.clear();
    full_img_sizes.clear();
    pairwise_matches.clear();
    cameras.clear();
    indices.clear();
}

void frameStitch::getImgNames(vector<String> imgNames){
    img_names.clear();
    for (auto i = imgNames.begin(); i != imgNames.end(); ++i) {
        img_names.push_back(*i);
    }
    num_images = static_cast<int>(img_names.size());
}

bool frameStitch::resizeImages(){
    bool m_status = true;

    images.clear();
    full_images.clear();
    full_img_sizes.clear();
    
    Mat full_img, img;

    for (int i = 0; i < num_images; ++i) {
        full_img = imread(img_names[i]);
        full_img_sizes.push_back(full_img.size());
        if (full_img.empty())
        {
            cout << "Can't open image " << img_names[i] << endl;
            m_status = false;
            return m_status;
        }
        
        if (!is_work_scale_set)
        {
            work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
            is_work_scale_set = true;
        }
        resize(full_img, img, Size(), work_scale, work_scale);
        full_images.push_back(img.clone());
        
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }
        
        resize(full_img, img, Size(), seam_scale, seam_scale);
        images.push_back(img.clone());
    }
    full_img.release();
    img.release();
    
    return m_status;
}

void frameStitch::findFeatures(){
    
    features.clear();
    
//    FileStorage fs("features.yml", FileStorage::WRITE);
    
    Ptr<FeaturesFinder> finder;
    finder = makePtr<SurfFeaturesFinder>();
    
    vector<ImageFeatures> local_features(num_images);
    
    
    for (int i = 0; i < num_images; ++i)
    {
        /* init features */
        (*finder)(full_images[i], local_features[i]);
        local_features[i].img_idx = i;
        cout << "Features in image #" << i+1 << ": " << local_features[i].keypoints.size() << endl;
    }
    features = local_features;
//    fs.release();
    finder->collectGarbage();
//    full_images.clear();
}

void frameStitch::findFrameFeature(string frameName){
    cout << "frame feature finding" << endl;
    Mat img;
    Mat img_gray = imread(frameName, IMREAD_GRAYSCALE);
    Mat img_color = imread(frameName);
    Ptr<FeaturesFinder> finder;
    finder = makePtr<SurfFeaturesFinder>();
    
    full_img_sizes.push_back(img_gray.size());
    
    resize(img_color, img, Size(), seam_scale, seam_scale);
    images.push_back(img.clone());
    
    resize(img_color, img, Size(), work_scale, work_scale);
    img_color = img.clone();
    full_images.push_back(img_color);
    
    
    (*finder)(img_color, frameFeature);
    
    frameFeature.img_idx = num_images;
    
    cout << "there are " << frameFeature.keypoints.size() << " features in this frame" << endl;
    
    finder->collectGarbage();
}

void frameStitch::pairwiseMatch(){
    cout << "pairwise matching" << endl;
    pairwise_matches.clear();
    indices.clear();
    
    if (range_width==-1)
    {
        BestOf2NearestMatcher matcher(try_cuda, match_conf);
        matcher(features, pairwise_matches);
        matcher.collectGarbage();
    }
    else
    {
        BestOf2NearestRangeMatcher matcher(range_width, try_cuda, match_conf);
        matcher(features, pairwise_matches);
        matcher.collectGarbage();
    }
    
    // Leave only images we are sure are from the same panorama
    indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }
    
    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;
    
    /* debug only */
//    for (auto i = 0; i < img_names.size(); ++i) {
//        cout << img_names[i] << endl;
//    }
}

void frameStitch::pairwiseFrameMatch(string frame_name){
    cout << "pairwsie frame matching" << endl;
    
    pairwise_frame_matches.clear();
    
    BestOf2NearestMatcher matcher(try_cuda, match_conf);
    allFeatures = features;
    allFeatures.push_back(frameFeature);
    matcher(allFeatures, pairwise_frame_matches);
    
    /////////////////// only compute 3 parameters
    threeFeatures.push_back(allFeatures[0]);
    threeFeatures.push_back(allFeatures[7]);
    threeFeatures.push_back(allFeatures[8]);
    matcher(threeFeatures, pairwise_frame_matches_three);
    vector<String> tmpImageNames{img_names[0], img_names[7], frame_name};
    cout << matchesGraphAsString(tmpImageNames, pairwise_frame_matches_three, conf_thresh) << endl;
    ////////////////// end 3 parameter
    
    matcher.collectGarbage();
    
    img_names.push_back(frame_name);
//    cout << matchesGraphAsString(img_names, pairwise_frame_matches, conf_thresh) << endl;
    
}

bool frameStitch::computeCameras(int sceneid){
    cout << "compute cameras" << endl;
    cameras.clear();
    bool b_status = true;
    
    HomographyBasedEstimator estimator;
    if (!estimator(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
        b_status = false;
        return b_status;
    }
    
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        cout << "Initial intrinsics #" << indices[i]+1 << ":\n" << "K" << ":\n" << cameras[i].K() << "\n" << "R" << ":\n" << cameras[i].R << endl;
    }
    
    
    /* write the camera params out */
    FileStorage fs("cameras_ori_" + to_string(sceneid) + ".yml", FileStorage::WRITE);
    for(auto i = 0; i < cameras.size(); ++i){
        fs << "camera" + to_string(i) << cameras[i];
    }
    fs.release();
    /////////////////////// end write files
    
//    Ptr<BundleAdjusterBaseRefine> adjuster;
    Ptr<detail::BundleAdjusterBase> adjuster;
    
//    if (ba_cost_func == "reproj") adjuster = makePtr<BundleAdjusterReprojRefine>();
//    else if (ba_cost_func == "ray") adjuster = makePtr<BundleAdjusterRayRefine>();
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    
    
    
    if (!(*adjuster)(features, pairwise_matches, cameras))
    /* change the adjust to my own */
    {
        cout << "Camera parameters adjusting failed.\n";
        b_status = false;
        return b_status;
    }
    
    /* debug only */
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        
//        cout << "Modified intrinsics #" << indices[i]+1 << ":\n" << "K" << ":\n" << cameras[i].K() << "\n" << "R" << ":\n" << cameras[i].R << "\n" <<endl;
    }
    
    
    
    return b_status;
}

void frameStitch::computeFrameCameraPos(){
    cout << "compute frame camera pos" << endl;
    int maxh_pair_idx = -1;
    double tmpConf = -1.0;
    int tmp_frame_idx = -1;
    int tmp_image_idx = -1;
    
    /* find out the most matching image with the frame */
    for (auto i = 0; i < pairwise_frame_matches.size(); ++i) {
        if (pairwise_frame_matches[i].src_img_idx == allFeatures.size() - 1) {
//            cout << "homo matrix is: \n" << pairwise_frame_matches[i].H << "\n" <<endl;
            tmp_frame_idx = pairwise_frame_matches[i].src_img_idx;
            tmp_image_idx = pairwise_frame_matches[i].dst_img_idx;
            cout << "the current i is: " << i << endl;
            cout << "src image index: " << tmp_frame_idx << endl;
            cout << "the des image index: " << tmp_image_idx << endl;
            cout << "the confidence is : " << pairwise_frame_matches[i].confidence << endl;
            
            if (pairwise_frame_matches[i].confidence > tmpConf) {
                tmpConf = pairwise_frame_matches[i].confidence;
                maxh_pair_idx = i;
            }
        }
    }
    ////////
//    maxh_pair_idx = 73;
    ////////
    tmp_frame_idx = pairwise_frame_matches[maxh_pair_idx].src_img_idx;
    tmp_image_idx = pairwise_frame_matches[maxh_pair_idx].dst_img_idx;
    cout << "src image index: " << tmp_frame_idx << endl;
    cout << "the max des image index: " << tmp_image_idx << endl;
    cout << "the confidence is : " << pairwise_frame_matches[maxh_pair_idx].confidence << endl;
    
    
    /* compute the camera matrix through homo matrix
    // x1 = K1 * R1 * R0.INV * K0.INV * x0
    // x1 = H01 * x0
    // K0 * R0 = H01.INV * K1 * R1
    // R0 = K.INV * H01.INV * K * R1
    */
    Mat_<double> K_img = Mat::eye(3, 3, CV_64F);
    K_img(0,0) = cameras[tmp_image_idx].focal;
    K_img(1,1) = cameras[tmp_image_idx].focal * cameras[tmp_image_idx].aspect;
    K_img(0,2) = cameras[tmp_image_idx].ppx;
    K_img(1,2) = cameras[tmp_image_idx].ppy;
    
    
    //////
    K_img = cameras[tmp_frame_idx].K();
    //////
    
    Mat KR_frame = pairwise_frame_matches[maxh_pair_idx].H.inv() * K_img;
    Mat tmpR, tmpKR, tmpK;
    cameras[tmp_image_idx].R.convertTo(tmpR, CV_64F);
    KR_frame.convertTo(tmpKR, CV_64F);
    Mat R_frame = K_img.inv() * tmpKR * tmpR;
    R_frame.convertTo(tmpR, CV_32F);
    K_img.convertTo(tmpK, CV_32F);
    
    cout << "the updated frame rotation is:\n" << tmpR << endl;
    
    cout << "the K matrix is:\n" << tmpK << endl;

    /* set the images' K to frame's K */
    cameras[cameras.size() - 1].focal = tmpK.at<float>(0,0);
    cameras[cameras.size() - 1].aspect = tmpK.at<float>(1,1) / tmpK.at<float>(0,0);
    cameras[cameras.size() - 1].ppx = tmpK.at<float>(0,2);
    cameras[cameras.size() - 1].ppy = tmpK.at<float>(1,2);
    
    /* set the images' R to frame's R */
//    cameras[cameras.size() - 1].R = cameras[tmp_image_idx].R;
    
    /* set the homo computed R to frame's R */
    cameras[cameras.size() - 1].R = tmpR;
    
    
    /* debug only */
    for (auto i = 0; i < cameras.size(); ++i) {
                if(i == tmp_image_idx || i == tmp_frame_idx) {
            cout << "homo compute camera#" << i << ":\n" << "K" << ":\n" << cameras[i].K() << "\n" << "R" << ":\n" << cameras[i].R << "\n" <<endl;
        }
    }
}

void frameStitch::getPairKeypoints(){
    cout << "get pair key points" << endl;
    
    vector<Point2f> frame2fpoints;
    vector<ImageKeypoint> imgKeyPoints;
    int m_scale = 100;
    
    int m_count = 1;
    
    /* 1. find out which image the feature belong to */
    for (auto i = 0; i < frameFeature.keypoints.size() / m_scale; ++i) {
        /* 1. find out which image the feature belong to */
        auto kp_pt = frameFeature.keypoints[i].pt;
        
        Mat kp_f_d = frameFeature.descriptors.getMat(ACCESS_READ).row(i);
        int num_options = 5;
        
        float minMatchDist = 100.f;
        int maxMatchNum = 0;
        int minMatchDist_sceneIdx = -1;
        KeyPoint bestMatchKp;
        
        // find 5 best match keypoints in each image
        for (auto sceneIdx = 0; sceneIdx < features.size(); sceneIdx++) {
            
            Mat i_d = features[sceneIdx].descriptors.getMat(ACCESS_READ);
            vector<pair<float, int> > options; options.clear();
            
            // find 5 best match key point
            for (auto j = 0; j < i_d.rows; j++) {
                Mat kp_i_d = i_d.row(j);
                float dist = norm(kp_f_d, kp_i_d);
                if (options.size() < num_options) {
                    options.push_back(pair<float, int>(dist, j));
                    sort(options.begin(), options.end());
                }
                else {
                    pair<float, int>* tail = &options[num_options - 1];
                    if (dist < tail->first) {
                        tail->first = dist;
                        tail->second = j;
                        sort(options.begin(), options.end());
                    }
                }
            }
            
            //            Mat img_clone = img.clone();
            //            circle(img_clone, kp_pt, 5, Scalar(0,0,255));
            //            circle(img_clone, kp_pt, 50, Scalar(0,0,255));
            //            Mat scene = full_images[sceneIdx].clone();
            //            for (auto j = 0; j < num_options; j++) {
            //                circle(scene, features[sceneIdx].keypoints[options[j].second].pt, 50, Scalar(0,0,51 * j));
            //            }
            //
            //            imshow("frame", img_clone);
            //            imshow("image", scene);
            //            for (auto j = 0; j < num_options; j++) {
            //                cout << options[j].first << " ";
            //            }
            //            cout << endl;
            //            waitKey(0);
            
            //find these keypoints whos distance is less than 50, the centor is the keypoint
            vector<int> kp_n_i, kp_n_f;
            float neighbor_dist = 1000.f;
            for (auto j = 0; j < frameFeature.keypoints.size(); j++) {
                auto kp_j_pt = frameFeature.keypoints[j].pt;
                float dist = (kp_pt.x - kp_j_pt.x) * (kp_pt.x - kp_j_pt.x) +
                (kp_pt.y - kp_j_pt.y) * (kp_pt.y - kp_j_pt.y);
                if (dist < neighbor_dist) {
                    kp_n_f.push_back(j);
                }
            }
            
            // copy those fit keypoints' descriptor to a matrix
            Mat kp_n_f_d;
            kp_n_f_d.create((int)kp_n_f.size(), kp_f_d.cols, kp_f_d.type());
            Mat f_d = frameFeature.descriptors.getMat(ACCESS_READ);
            for (auto j = 0; j < kp_n_f.size(); j++) {
                f_d.row(kp_n_f[j]).copyTo(kp_n_f_d.row(j));
            }
            
            // find the min avg distance for those 5 options, and the R is 75, which is a little larger than 50
            float neighbor_dist_option = 1.5 * neighbor_dist;
            for (auto optionIdx = 0; optionIdx < num_options; optionIdx++) {
                kp_n_i.clear();
                auto kp_option_pt = features[sceneIdx].keypoints[options[optionIdx].second].pt;
                for (auto j = 0; j < features[sceneIdx].keypoints.size(); j++) {
                    auto kp_i_pt = features[sceneIdx].keypoints[j].pt;
                    float dist = (kp_i_pt.x - kp_option_pt.x) * (kp_i_pt.x - kp_option_pt.x) +
                    (kp_i_pt.y - kp_option_pt.y) * (kp_i_pt.y - kp_option_pt.y);
                    if (dist < neighbor_dist_option) {
                        kp_n_i.push_back(j);
                    }
                }
                Mat kp_n_i_d, i_d;
                i_d = features[sceneIdx].descriptors.getMat(ACCESS_READ);
                kp_n_i_d.create((int)kp_n_i.size(), i_d.cols, i_d.type());
                
                for (auto j = 0; j < kp_n_i.size(); j++) {
                    i_d.row(kp_n_i[j]).copyTo(kp_n_i_d.row(j));
                }
                
                FlannBasedMatcher matcher;
                vector<DMatch> matches;
                matcher.match(kp_n_f_d, kp_n_i_d, matches);
                
                // count avg distance
                float avgDist = 0;
                for (auto j = 0; j < matches.size(); j++) {
                    avgDist += matches[j].distance;
                }
                
                avgDist /= matches.size();
                
                if (avgDist < minMatchDist) {
//                    cout << "update" << avgDist << " " << minMatchDist << endl  ;
                    minMatchDist = avgDist;
                    minMatchDist_sceneIdx = sceneIdx;
                    bestMatchKp = features[sceneIdx].keypoints[options[optionIdx].second];
                }
                // cout max match num
//                if (matches.size() > maxMatchNum) {
//                    maxMatchNum = int(matches.size());
//                    minMatchDist_sceneIdx = sceneIdx;
//                    bestMatchKp = features[sceneIdx].keypoints[options[optionIdx].second];
//                    
//                }
                
//                cout << optionIdx << " " << avgDist << " " << minMatchDist << endl;
                
            }
            
        }
        
        ImageKeypoint ikp;
        ikp.img_idx = minMatchDist_sceneIdx;
        ikp.keypoint = bestMatchKp;
        /* find out the 2d point of this keypoint */
        ikp.frameUV = kp_pt;
        
        imgKeyPoints.push_back(ikp);
        
        
        //        //DEBUG purpose
//        Mat img;
//        Mat frameImg = imread("/Volumes/Transcend/pro/opencv/data/02443.jpg");
//        resize(frameImg, img, Size(), work_scale, work_scale);
//        
//        Mat img_clone = img.clone();
//        circle(img_clone, kp_pt, 10, Scalar(0,0,255), -1);
//        Mat scene = full_images[ikp.img_idx].clone();
//        circle(scene, ikp.keypoint.pt, 10, Scalar(0, 0, 255),-1);
//
//        cout << full_images.size() << " " << ikp.img_idx << " " << frameFeature.keypoints[i].octave << " " <<frameFeature.keypoints[i].response <<  endl;
        
        // write image files
//        string tmpfilename = "tmp/" + to_string(m_count) + "a" + ".jpg";
//        imwrite(tmpfilename.c_str(), img_clone);
//        tmpfilename = "tmp/" + to_string(m_count) + "b" + ".jpg";
//        imwrite(tmpfilename.c_str(), scene);
//        m_count++;
        
        // show image files
//        imshow("frame", img_clone);
//        imshow("image", scene);
//
//        waitKey(0);
    }
    g_imgKeypoint = imgKeyPoints;
    
    //DEBUG PURPOSE
    for (auto i = 0; i < g_imgKeypoint.size(); ++i) {
        cout << "this keypoint is belong to image: " << g_imgKeypoint[i].img_idx << endl;
        cout << "and the image UV is: " << g_imgKeypoint[i].keypoint.pt << endl;
        cout << "and the frame UV is: " << g_imgKeypoint[i].frameUV << endl;
    }
    
}

void frameStitch::computeFrameCameraPosV2(){
    cout << "compute frame camera version 2" << endl;
    
    vector<vector<Point3f>> worldCoord;
    vector<vector<Point2f>> imgCoord;
    vector<Point2f> frame2fpoints;
    vector<Point3f> obj3Dpoints;
    vector<ImageKeypoint> imgKeyPoints;

    Mat K_frame, R_frame, T_frame, D_frame;
    
//    DEBUG
//    Mat drawImg;
    Mat img;
    Mat frameImg = imread("/Volumes/Transcend/pro/opencv/data/01509b.jpg");
    resize(frameImg, img, Size(), work_scale, work_scale);
    
    /* 1. find out which image the feature belong to */
    for (auto i = 0; i < frameFeature.keypoints.size(); ++i) {
        /* 1. find out which image the feature belong to */
        auto kp_pt = frameFeature.keypoints[i].pt;
        
        /* 2. find out the 2d point of this keypoint */
        frame2fpoints.push_back(kp_pt);
        // end 2
        
        Mat kp_f_d = frameFeature.descriptors.getMat(ACCESS_READ).row(i);
        int num_options = 5;
        
        float minMatchDist = 100.f;
        int minMatchDist_sceneIdx = -1;
        KeyPoint bestMatchKp;
        
        for (auto sceneIdx = 0; sceneIdx < features.size(); sceneIdx++) {
            
            Mat i_d = features[sceneIdx].descriptors.getMat(ACCESS_READ);
            vector<pair<float, int> > options; options.clear();
            for (auto j = 0; j < i_d.rows; j++) {
                Mat kp_i_d = i_d.row(j);
                float dist = norm(kp_f_d, kp_i_d);
                if (options.size() < num_options) {
                    options.push_back(pair<float, int>(dist, j));
                    sort(options.begin(), options.end());
                }
                else {
                    pair<float, int>* tail = &options[num_options - 1];
                    if (dist < tail->first) {
                        tail->first = dist;
                        tail->second = j;
                        sort(options.begin(), options.end());
                    }
                }
            }
            
//            Mat img_clone = img.clone();
//            circle(img_clone, kp_pt, 5, Scalar(0,0,255));
//            circle(img_clone, kp_pt, 50, Scalar(0,0,255));
//            Mat scene = full_images[sceneIdx].clone();
//            for (auto j = 0; j < num_options; j++) {
//                circle(scene, features[sceneIdx].keypoints[options[j].second].pt, 50, Scalar(0,0,51 * j));
//            }
//    
//            imshow("frame", img_clone);
//            imshow("image", scene);
//            for (auto j = 0; j < num_options; j++) {
//                cout << options[j].first << " ";
//            }
//            cout << endl;
//            waitKey(0);

            
            vector<int> kp_n_i, kp_n_f;
            float neighbor_dist = 50.f;
            for (auto j = 0; j < frameFeature.keypoints.size(); j++) {
                auto kp_j_pt = frameFeature.keypoints[j].pt;
                float dist = (kp_pt.x - kp_j_pt.x) * (kp_pt.x - kp_j_pt.x) +
                (kp_pt.y - kp_j_pt.y) * (kp_pt.y - kp_j_pt.y);
                if (dist < neighbor_dist) {
                    kp_n_f.push_back(j);
                }
            }
            
            Mat kp_n_f_d;
            kp_n_f_d.create((int)kp_n_f.size(), kp_f_d.cols, kp_f_d.type());
            Mat f_d = frameFeature.descriptors.getMat(ACCESS_READ);
            for (auto j = 0; j < kp_n_f.size(); j++) {
                f_d.row(kp_n_f[j]).copyTo(kp_n_f_d.row(j));
            }
            
            float neighbor_dist_option = 75.f;
            for (auto optionIdx = 0; optionIdx < num_options; optionIdx++) {
                kp_n_i.clear();
                auto kp_option_pt = features[sceneIdx].keypoints[options[optionIdx].second].pt;
                for (auto j = 0; j < features[sceneIdx].keypoints.size(); j++) {
                    auto kp_i_pt = features[sceneIdx].keypoints[j].pt;
                    float dist = (kp_i_pt.x - kp_option_pt.x) * (kp_i_pt.x - kp_option_pt.x) +
                    (kp_i_pt.y - kp_option_pt.y) * (kp_i_pt.y - kp_option_pt.y);
                    if (dist < neighbor_dist_option) {
                        kp_n_i.push_back(j);
                    }
                }
                Mat kp_n_i_d, i_d;
                i_d = features[sceneIdx].descriptors.getMat(ACCESS_READ);
                kp_n_i_d.create((int)kp_n_i.size(), i_d.cols, i_d.type());
                
                for (auto j = 0; j < kp_n_i.size(); j++) {
                    i_d.row(kp_n_i[j]).copyTo(kp_n_i_d.row(j));
                }
                
                FlannBasedMatcher matcher;
                vector<DMatch> matches;
                matcher.match(kp_n_f_d, kp_n_i_d, matches);
                
                float avgDist = 0;
                for (auto j = 0; j < matches.size(); j++) {
                    avgDist += matches[j].distance;
                }
                
                avgDist /= matches.size();
                
                if (avgDist < minMatchDist) {
                    cout << "update" << avgDist << " " << minMatchDist << endl  ;
                    minMatchDist = avgDist;
                    minMatchDist_sceneIdx = sceneIdx;
                    bestMatchKp = features[sceneIdx].keypoints[options[optionIdx].second];
                }
                
                cout << optionIdx << " " << avgDist << " " << minMatchDist << endl;
                
            }
            
        }
        
        ImageKeypoint ikp;
        ikp.img_idx = minMatchDist_sceneIdx;
        ikp.keypoint = bestMatchKp;
        
        imgKeyPoints.push_back(ikp);

        
//        //DEBUG purpose
        
//        Mat img_clone = img.clone();
//        circle(img_clone, kp_pt, 50, Scalar(0,0,255));
//        Mat scene = full_images[ikp.img_idx].clone();
//        circle(scene, ikp.keypoint.pt, 50, Scalar(0, 0, 255));

//        cout << full_images.size() << " " << ikp.img_idx << " " << frameFeature.keypoints[i].octave << " " <<frameFeature.keypoints[i].response <<  endl;
//        
//        imshow("frame", img_clone);
//        imshow("image", scene);
//
//        waitKey(0);
    }

    
    /* 3. compute the 3d point of this keypoint */
    for (auto it = imgKeyPoints.begin(); it != imgKeyPoints.end(); ++it) {
        // get the corresponding camera intrincs
        Mat K_img, tmpK;
        Mat R_img, tmpR, tmpX3d;
        Mat_<double> x2d_img = Mat::eye(3, 1, CV_64F);
        Mat_<double> x3d_obj = Mat::eye(3, 1, CV_64F);
        Point3f point3f_obj;
        
        K_img = cameras[(*it).img_idx].K();
        K_img.convertTo(tmpK, CV_64F);
        R_img = cameras[(*it).img_idx].R;
        R_img.convertTo(tmpR, CV_64F);
        
        x2d_img(0,0) = (*it).keypoint.pt.x;
        x2d_img(1,0) = (*it).keypoint.pt.y;
        x2d_img(2,0) = 1;
        
        // x3d = R.inv * k.inv * x2d
        x3d_obj = tmpR.inv() * tmpK.inv() * x2d_img;
        x3d_obj.convertTo(tmpX3d, CV_32F);
        
        // get value from matrix(3,1) to 3f point
        point3f_obj.x = x3d_obj.at<float>(0,0);
        point3f_obj.y = x3d_obj.at<float>(1,0);
        point3f_obj.z = x3d_obj.at<float>(2,0);
        
        obj3Dpoints.push_back(point3f_obj);
        
        //DEBUG PURPOSE
//        cout << "3d point is: " << point3f_obj << endl;
        
        
        
//        DEBUG draw these keypoints on the frame
//        cout << "The coord on the frame is: " << frameFeature.keypoints[i].pt << endl;
//        drawKeypoints(img, frameFeature.keypoints, drawImg);
//        imshow("drawFeatures", drawImg);
//        waitKey(0);
        
    }
    
    /* read K params from other camearas */
//    cameras[0].K().convertTo(K_frame, CV_32F);
    K_frame = cameras[0].K();
    D_frame = Mat::zeros(4, 1, CV_64FC1);
    
    ////////////
    vector<Point3f> new_o_3d;
    vector<Point2f> new_f_2f;
    int m_scale = 1;
    for (auto i = 0; i < obj3Dpoints.size()/m_scale; ++i) {
        cout << "3d points is: " << obj3Dpoints[i] << endl;
        new_o_3d.push_back(obj3Dpoints[i]);
    }
    cout << "\n\n\n" << endl;
    
    for (auto i = 0; i < frame2fpoints.size()/m_scale; ++i) {
        cout << "2d points are: " << frame2fpoints[i] << endl;
        new_f_2f.push_back(frame2fpoints[i]);
    }
    ////////////
    
    /* 4. set these 3d and 2d point to camera calibration to compute the camera intrics */
    cout << "the frame size is: " << full_images[num_images - 1].size() << endl;
    cout << "the size of 3d is: " << obj3Dpoints.size() << endl;
    cout << "the size of 2d is: " << frame2fpoints.size() << endl;
    cout << "there are " << obj3Dpoints.size() / m_scale << " points in count" << endl;
//    worldCoord.push_back(new_o_3d);
//    imgCoord.push_back(new_f_2f);
//    calibrateCamera(worldCoord, imgCoord, full_images[num_images - 1].size(), K_frame, D_frame, R_frame, T_frame, CV_CALIB_USE_INTRINSIC_GUESS);
    cout << "the matrix of K is: \n" << K_frame << endl;

    solvePnPRansac(new_o_3d, new_f_2f, K_frame, D_frame, R_frame, T_frame);
    
    Mat R_3x3_frame, R_3x3_tmp;
    Rodrigues(R_frame, R_3x3_frame);
    R_3x3_frame.convertTo(R_3x3_tmp, CV_32F);
    
    //DEBUG ONLY
    cout << "the matrix of R is: \n" << R_3x3_tmp << endl;
    
    /* set K and R to the frame camera */
    cameras[cameras.size() - 1].focal = K_frame.at<float>(0,0);
    cameras[cameras.size() - 1].aspect = K_frame.at<float>(1,1) / K_frame.at<float>(0,0);
    cameras[cameras.size() - 1].ppx = K_frame.at<float>(0,2);
    cameras[cameras.size() - 1].ppy = K_frame.at<float>(1,2);
    
    cameras[cameras.size() - 1].R = R_3x3_tmp;
}

bool frameStitch::computeFrameCamera(){
    cout << "compute frame camera" << endl;
    bool b_status = true;
    
//    CameraParams frame_cam;
//    cameras.push_back(frame_cam);
    
//    for (auto i = 0; i < cameras.size(); ++i) {
//        Mat R;
//        cameras[i].R.convertTo(R, CV_64FC1);
//        cameras[i].R = R;
//        
//        /* debug only */
////                if(i == allCamParams.size() - 1 || i == allCamParams.size() - 2) {
////                    cout << "new generated intrinsics #" << i << ":\n" << "K" << ":\n" << cameras[i].K() << "\n" << "R" << ":\n" << cameras[i].R << "\n" <<endl;
////                }
//    }
    
    vector<CameraParams> allCamParams;
    
    HomographyBasedEstimatorRefine estimator;
    
    estimator(allFeatures, pairwise_frame_matches, allCamParams);
    
//    estimator(allFeatures, pairwise_frame_matches, cameras);
    
    
    /* debug only */
    for (auto i = 0; i < allCamParams.size(); ++i) {
        Mat R;
        allCamParams[i].R.convertTo(R, CV_32F);
        allCamParams[i].R = R;
        
        /* debug only */
        cout << "new generated intrinsics #" << i << ":\n" << "K" << ":\n" << allCamParams[i].K() << "\n" << "R" << ":\n" << allCamParams[i].R << "\n" <<endl;
    }
    
    cameras = allCamParams;
    num_images  += 1;
    
    return b_status;
}

bool frameStitch::computeFrameBundleAdjustment(){
    cout << "compute frame camera bundle adjustment" << endl;
    bool b_status = true;
    
    Ptr<BundleAdjusterBaseRefine> adjuster;
    //    Ptr<detail::BundleAdjusterBase> adjuster;
    
    if (ba_cost_func == "reproj") adjuster = makePtr<BundleAdjusterReprojRefine>();
    else if (ba_cost_func == "ray") adjuster = makePtr<BundleAdjusterRayRefine>();
    //    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    //    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    
    /* fix pano camera params */
    //    for (auto i = 0; i < num_images; ++i) {
    //        allCamParams[i] = cameras[i];
    //    }
    
    // only contain 3 camera parameters
    vector<CameraParams> tmpCameras;
    tmpCameras.push_back(cameras[0]);
    tmpCameras.push_back(cameras[7]);
    tmpCameras.push_back(cameras[8]);
    
    int64 t = getTickCount();
    
    if (!(*adjuster)(allFeatures, pairwise_frame_matches, cameras))
        //        if(!(*adjuster)(features, pairwise_matches, cameras))
    /* change the adjust to my own */
    {
        cout << "Camera parameters adjusting failed.\n";
        b_status = false;
        return b_status;
    }
    cout << "BundleAdjustment all, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

    
    ///////////////////////////////////////////////////////////////// only compute 3 parameters
    t = getTickCount();
    if (!(*adjuster)(threeFeatures, pairwise_frame_matches_three, tmpCameras))
        //        if(!(*adjuster)(features, pairwise_matches, cameras))
    /* change the adjust to my own */
    {
        cout << "Camera parameters adjusting failed.\n";
        b_status = false;
        return b_status;
    }
    cout << "BundleAdjustment three, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
    
//    cameras[8] = tmpCameras[2];
    
    ///////////////////////////////////////////////////////////////// end only compute 3 parameters
    
    return b_status;
}

void frameStitch::waveCorrectionFrame(){
    
    /* do wave correction */
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        //        for (size_t i = 0; i < cameras.size(); ++i){
        cameras[cameras.size() - 1].R = rmats[cameras.size() - 1];
        //        }
    }
    
    /* rotate pitch */
//    float y_angle_d = -0.635;
//    
//    Mat R_y = (Mat_<float>(3,3) <<
//               cos(y_angle_d),  0, sin(y_angle_d),
//               0             ,  1, 0             ,
//               -sin(y_angle_d), 0, cos(y_angle_d)
//               );
//    
////    Mat R_z = (Mat_<double>(3,3) <<
////               cos(z_angle_d), -sin(z_angle_d), 0,
////               sin(z_angle_d),  cos(z_angle_d), 0,
////               0             ,  0             , 1
////               );
//
//    /* rotate the frame camera by -30 degree */
//    cameras[cameras.size() - 1].R *= R_y;
    //////////////rotate pitch end
    
    
}

void frameStitch::waveCorrection(int sceneid){
    // Find median focal length
    cout << "start wave correction" << endl;
    
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        focals.push_back(cameras[i].focal);
    }
    
    sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
    
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i){
            cameras[i].R = rmats[i];
//            cout << "Wave correction intrinsics #" << indices[i]+1 << ":\n" << "K" << ":\n" << cameras[i].K() << "\n" << "R" << ":\n" << cameras[i].R << endl;
        }
    }
    
    /* write the value of the focal */
    FileStorage fs_focal("focal_" + to_string(sceneid) + ".yml", FileStorage::WRITE);
    fs_focal << "focal" << warped_image_scale;
    fs_focal.release();
    
    /* write the number of selected images */
    FileStorage fs_num("selected_" + to_string(sceneid) + ".yml", FileStorage::WRITE);
    fs_num << "image_num" << indices;
    fs_num.release();
    
    /* write the camera params out */
    FileStorage fs("cameras_" + to_string(sceneid) + ".yml", FileStorage::WRITE);
    for(auto i = 0; i < cameras.size(); ++i){
        fs << "camera" + to_string(i) << cameras[i];
    }
    fs.release();
}

void frameStitch::readCameras(int group_idx){
    cout << "read camera parameters from file" << endl;
    
    FileStorage fs_focal("focal_" + to_string(group_idx) + ".yml", FileStorage::READ);
    fs_focal["focal"] >> warped_image_scale;
    fs_focal.release();
    
    FileStorage fs_num("selected_" + to_string(group_idx) + ".yml", FileStorage::READ);
    fs_num["image_num"] >> indices;
    fs_num.release();
    
    FileStorage fs("cameras_" + to_string(group_idx) + ".yml", FileStorage::READ);
    for (auto i = 0; i < indices.size(); ++i) {
        string tmpName = "camera" + to_string(i);
        CameraParams tmpCam;
        fs[tmpName] >> tmpCam;
        cameras[i] = tmpCam;
    }
    fs.release();
    
    /* debug only */
//    for (size_t i = 0; i < cameras.size(); ++i){
//        cout << "read camera intrinsics #" << indices[i]+1 << ":\n" << "K" << ":\n" << cameras[i].K() << "\n" << "R" << ":\n" << cameras[i].R << endl;
//    }
}

void frameStitch::readCamerasOri(int group_idx){
    cout << "read camera parameters from file" << endl;
//    cameras.clear();
    
    FileStorage fs_focal("focal_" + to_string(group_idx) + ".yml", FileStorage::READ);
    fs_focal["focal"] >> warped_image_scale;
    fs_focal.release();
    
    FileStorage fs_num("selected_" + to_string(group_idx) + ".yml", FileStorage::READ);
    fs_num["image_num"] >> indices;
    fs_num.release();
    
    FileStorage fs("cameras_ori_" + to_string(group_idx) + ".yml", FileStorage::READ);
    for (auto i = 0; i < indices.size(); ++i) {
        string tmpName = "camera" + to_string(i);
        CameraParams tmpCam;
        fs[tmpName] >> tmpCam;
        cameras[i] = tmpCam;
//        cameras.push_back(tmpCam);
    }
    fs.release();
    
    /* debug only */
    //    for (size_t i = 0; i < cameras.size(); ++i){
    //        cout << "read camera intrinsics #" << indices[i]+1 << ":\n" << "K" << ":\n" << cameras[i].K() << "\n" << "R" << ":\n" << cameras[i].R << endl;
    //    }
}


bool frameStitch::warpImages(){
    cout << "warping images" << endl;
    bool b_status = true;
    
    vector<Point> local_corners(num_images);
    vector<UMat> local_masks_warped(num_images);
    vector<UMat> local_images_warped(num_images);
    vector<Size> local_sizes(num_images);
    vector<UMat> masks(num_images);
    
    // Preapre images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }
    
    // Warp images and their masks
    
    warper_creator = makePtr<cv::SphericalWarper>();

    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        b_status = false;
        return b_status;
    }

    warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {        
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;
        
        local_corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, local_images_warped[i]);
        local_sizes[i] = local_images_warped[i].size();
        
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, local_masks_warped[i]);
    }
    
    corners = local_corners;
    images_warped = local_images_warped;
    masks_warped = local_masks_warped;
    sizes = local_sizes;
    
    return b_status;
}

void frameStitch::compensateExposure(){
    cout << "compensate exposure errors" << endl;
    
    compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);

}

void frameStitch::findSeamMask(){
#if (G_INITIAL==1)
    cout << "find seam mask" << endl;
    Ptr<SeamFinder> seam_finder;
    seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    
    seam_finder->find(images_warped_f, corners, masks_warped);

    images_warped_f.clear();
#endif
    images.clear();
    images_warped.clear();
}

/* split a string into x,y and store in a point */

void SplitString(const std::string& s, Point& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    if(std::string::npos != pos2)
    {
        /* asign x */
        v.x = stoi(s.substr(pos1, pos2-pos1));
        
        pos1 = pos2 + c.size();
        //        pos2 = s.find(c, pos1); //ori
        v.y = stoi(s.substr(pos1));
        /* asign y */
    }
    //    if(pos1 != s.length()) //ori
    //        v.push_back(s.substr(pos1)); //ori
}

void frameStitch::blendImages(int group_idx, bool initial, string inputImage){
    cout << "blend images" << endl;
    
    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    
    Mat full_img;
    Mat img;
    
    double compose_work_aspect = 1;
    
    /* define files for corners and sizes */
#if (G_INITIAL==1)
    ofstream corner_file_out, size_file_out;
    corner_file_out.open("corner_" + to_string(group_idx) + ".txt");
    size_file_out.open("size_" + to_string(group_idx) + ".txt");
#else
    ifstream corner_file_in, size_file_in;
    corner_file_in.open("corner_" + to_string(group_idx) + ".txt");
    size_file_in.open("size_" + to_string(group_idx) + ".txt");
#endif
    
#if (G_INITIAL==0)
    vector<Point> tmpCorners;
    vector<Size> tmpsizes;
    Point tmpCorner;
    Size tmpsize;
    string s_line;
    
    vector<Point> old_corners;
    /* read corners */
    if(corner_file_in.is_open()){
        while (getline(corner_file_in, s_line)) {
            SplitString(s_line, tmpCorner, " ");
            tmpCorners.push_back(tmpCorner);
            cout << tmpCorner << endl;
        }
    }
    tmpCorners.push_back(corners[num_images - 1]);
    corners = tmpCorners;
    old_corners = corners;
    
//    for (auto i = 0; i < corners.size(); ++i) {
//        cout << "debug corners: " <<corners[i].x << " " << corners[i].y << endl;
//    }

    vector<Size> old_sizes;
    /* read sizes */
    if(size_file_in.is_open()){
        while (getline(size_file_in, s_line)) {
            SplitString(s_line, tmpCorner, " ");
            tmpsize.width = tmpCorner.x;
            tmpsize.height = tmpCorner.y;
            tmpsizes.push_back(tmpsize);
            cout << tmpsize << endl;
        }
    }
    tmpsizes.push_back(sizes[num_images - 1]);
    sizes = tmpsizes;
    old_sizes = sizes;
    
    ////////////
    for (auto i = 0; i < corners.size(); ++i) {
        cout << "debug corners: " << corners[i].x << " " << corners[i].y << endl;
    }
    ////////////
#endif
    
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        cout << "Compositing image #" << img_idx << endl;
        
        // Read image and resize it if necessary
        full_img = imread(img_names[img_idx]);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;
            
            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;
            
            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);
            
            
            
            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;
                
                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }
                
                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                cout << "\n debug sz: " << sz << "\n" << endl;
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
#if (G_INITIAL==1)
                corners[i] = roi.tl();
                sizes[i] = roi.size();
#else
                if (i != num_images - 1){
                    old_corners[i] = roi.tl();
                    old_sizes[i] = roi.size();
                }
                else{
                    corners[i] = roi.tl();
                    sizes[i] = roi.size();
                }
#endif
                
                
#if (G_INITIAL==1)
                /* write corners and sizes into file */
                corner_file_out << corners[i].x << " " << corners[i].y << "\n";
                size_file_out << sizes[i].width << " " << sizes[i].height << "\n";
#endif
            }
            
#if (G_INITIAL==0)
            /* remap keypoints' coordinates */
            vector<int> m_offsets;
            for (auto i = 0; i < g_imgKeypoint.size(); ++i) {
                Mat tmp_K_i, tmp_K_f;
                int m_offset_x = 0;
                cameras[g_imgKeypoint[i].img_idx].K().convertTo(tmp_K_i, CV_32F);
                cameras[num_images - 1].K().convertTo(tmp_K_f, CV_32F);
//                g_imgKeypoint[i].keypoint.pt *= 1;
//                g_imgKeypoint[i].frameUV *= 1;
                
                Point2f kp_i_l = warper->warpPoint(g_imgKeypoint[i].keypoint.pt, tmp_K_i, cameras[g_imgKeypoint[i].img_idx].R);
                Point2f kp_f_l = warper->warpPoint(g_imgKeypoint[i].frameUV, tmp_K_f, cameras[num_images - 1].R);
                m_offset_x = kp_i_l.x - kp_f_l.x;
                
                cout << "the same feature' coord of image is: " << kp_i_l << endl;
                cout << "the same feature' coord of frame is: " << kp_f_l << endl;
                cout << "the image corner is: " << old_corners[g_imgKeypoint[i].img_idx] << endl;
                cout << "the frame corner is: " << corners[corners.size() - 1] << endl;
                cout << "and the offset is: " << m_offset_x << endl;
                
                m_offsets.push_back(m_offset_x);
            }
            sort(m_offsets.begin(), m_offsets.end());
            int final_offset_x = 0;
            int sum_offset_x = 0;
//            int final_offset_x = m_offsets[m_offsets.size() / 2];
            for (auto i = 0; i < 3; ++i) {
                sum_offset_x += m_offsets[m_offsets.size() / 2 - 2 + i];
            }
            final_offset_x = sum_offset_x / 3;
            
            float m_s = 0.25;
//            float m_s = 1;
            final_offset_x *= m_s;
            /////////
            /* debug only */
            
            for (auto i = 0; i < corners.size(); ++i) {
                cout << "debug corners: " << corners[i].x << " " << corners[i].y << endl;
            }
            corners[corners.size() - 1].x += final_offset_x;
//            corners[corners.size() - 1].y = 589;
            ////////
#endif
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();
        
        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);
        
        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        
        
        
        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        
//        imwrite("un" + to_string(img_idx) + "framemask.jpg", mask_warped);
        
        /* write final warped images and masks */
        string tempimagename = "final_warped_image_";
        tempimagename += to_string(group_idx);
        tempimagename += "_";
        tempimagename += to_string(img_idx);
        tempimagename += ".jpg";
#if (G_INITIAL == 1)
        imwrite(tempimagename, img_warped);
#else
        if(img_idx != num_images - 1)
            img_warped = imread(tempimagename);
#endif
        
        // Compensate exposure
//        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
        
        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();
#if (G_INITIAL == 0)
        /* add a circle to the frame */
        if(img_idx == num_images -1)
//            circle(img_warped_s, Point(img_warped_s.rows / 2, img_warped_s.cols / 2), 50, CvScalar(0,0,255), -1, 8);
//            rectangle(img_warped_s, Point(20, 30), Point(img_warped_s.cols - 20, img_warped_s.rows - 30), CvScalar(0,255,0), 20);
#endif
        
        
        
#if (G_INITIAL == 1)
        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;
#else
        /* the mask of the frame is all optical */
        if (img_idx != num_images - 1) {
            mask_warped = seam_mask & mask_warped;
        }
#endif
        
        // write the warped image out
        if (img_idx == num_images - 1) {
            string tmpName = "demo/warped.jpg";
            imwrite(tmpName, img_warped_s);
            string tmpMskName = "demo/mask.jpg";
            imwrite(tmpMskName, mask_warped);
        }
        
        string tempmaskname = "final_warped_mask_";
        tempmaskname += to_string(group_idx);
        tempmaskname += "_";
        tempmaskname += to_string(img_idx);
        tempmaskname += ".jpg";
#if (G_INITIAL == 1)
        imwrite(tempmaskname, mask_warped);
#else
        if(img_idx != num_images - 1)
            mask_warped = imread(tempmaskname, IMREAD_GRAYSCALE);
#endif
        
        if (!blender)
        {
            cout << "blender in" << endl;
            
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f/blend_width);
                cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
            }
            blender->prepare(corners, sizes);
        }
        

        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        
    }
    
#if (G_INITIAL==1)
    corner_file_out.close();
    size_file_out.close();
#else
    corner_file_in.close();
    size_file_in.close();
#endif
    
    Mat result, result_mask;
    cout << "blending..." << endl;
    blender->blend(result, result_mask);
    
    cout << "writing..." << endl;
#if (G_INITIAL == 1)
    imwrite(result_name + to_string(group_idx) + ".jpg", result);
#else
    string demo_result_name = "demo/result_";
    string name_plus;
    
    //find the last '/' in string
    auto m_found = inputImage.rfind("/");
    if (m_found != string::npos) {
        name_plus = inputImage.substr(m_found + 1, 5);
    }
    
    demo_result_name += name_plus;
    
    cout << demo_result_name << endl;
    cout << name_plus << endl;
    
    imwrite(demo_result_name + ".jpg", result);
//    imwrite(result_name + ".jpg", result);
#endif
    
}


