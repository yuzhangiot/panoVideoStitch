//
//  main.cpp
//  PhotoTourism
//
//  Created by Bichuan Guo on 5/29/16.
//  Copyright © 2016 loft. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "keypoint_filter.hpp"
#include "template_match.hpp"
#include "frameStitch.hpp"
#include "opencv2/stitching/detail/matchers.hpp"

#include <glob.h>

#define DEBUG_LEVEL_1 1
#define ROUGH_MATCH 1

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    
//    string fileLoc = "/Volumes/Transcend/pro/opencv/data/";
    string filetest = "00346.jpg";
    string fileInput = argv[1];

    
    bool initial = true;

    int numScenes;
    
    /*
     * read images in directory
     * Joseph
     * 2016.07.23
     */
    string root_dir = "/Volumes/Transcend/pro/opencv/data/testpanos/*";
//    string root_dir = "/Volumes/Transcend/pro/opencv/data/test/*";
    vector<vector<string>> all_images;
    
    glob_t glob_res;
    
    glob(root_dir.c_str(), GLOB_TILDE, NULL, &glob_res);
    
    for (auto i = 0; i < glob_res.gl_pathc; ++i) {
//        cout << glob_res.gl_pathv[i] << endl;
        vector<string> sub_images;
        glob_t glob_sub_res;
        string sub_dir = string(glob_res.gl_pathv[i]) + "/*";
        glob(sub_dir.c_str(), GLOB_TILDE, NULL, &glob_sub_res);
        for (auto j = 0; j < glob_sub_res.gl_pathc; ++j) {
            
            sub_images.push_back(glob_sub_res.gl_pathv[j]);
//            cout << glob_sub_res.gl_pathv[j] << endl;
        }
        all_images.push_back(sub_images);
    }
    
    numScenes = int(all_images.size());
//      DEBUG ONLY
//    for (auto i = 0; i < all_images.size(); ++i) {
//        for (auto j = 0; j < all_images[i].size(); ++j) {
//            cout << all_images[i][j] << endl;
//        }
//    }
//    
    
    
    ////////////////////////////////////////////////////////// first step start
    
    Mat query_img = imread(fileInput, IMREAD_GRAYSCALE);
    vector<KeyPoint> query_keypoints;
    Mat query_descriptor;
    
    //int minHessian = 400;
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    //detector->setHessianThreshold(minHessian);
    
    detector->detectAndCompute(query_img, Mat(), query_keypoints, query_descriptor);
    
    KeyPointFilter* filter = new KeyPointFilter;
    filter->create(numScenes);

    
    if (initial) {
        int tmpSum = 0;
        for (int i = 0; i < numScenes; i++) {
            if(i != 17) continue;
            for (int j = 0; j < all_images[i].size(); j++) {
//                sprintf(filename, "%d/q_pic%d.jpg", i + 1, j + 1);
                const Mat img = imread(all_images[i][j]);
#if ROUGH_MATCH == 1
                vector<KeyPoint> keypoints;
                Mat descriptor;
                detector->detectAndCompute(img, Mat(), keypoints, descriptor);
                
                cout << "image# " << j << " has: " << descriptor.rows << "descriptors" << endl;
                tmpSum += descriptor.rows;
//                std::cout << "add " << i << " " << j << std::endl;

                filter->addTrainSet(descriptor, i);
#elif ROUGH_MATCH == 2
//                singleGroup.push_back(filter->addTrainSetFull(img, j, int(all_images[i].size())));
#endif
            }
        }
        cout << "sum is: " << tmpSum << endl;
        return 0;

        
        filter->reduce();
        
        filter->save();
#if ROUGH_MATCH == 2
        for (auto i = 1; i <= 5; ++i) {
            filter->clfileToYml(i); // 1 is Symmetric, 2 is unsymmetric
        }
#endif
    }
    else {
//#if ROUGH_MATCH == 1
        // only load descriptor for 1st method
        filter->load();
//#elif ROUGH_MATCH == 2
        filter->loadcl(6);
//#endif
    }
    
//    filter->clustering();
    

    /* debug purpose */
//    for (int i = 0; i < filter->m_dict.size(); i++) {
//        std::cout << filter->m_dict[i].size() << std::endl;
//    }
    
    /* get the result from first step */
    vector<int> results_first;
    
//    cout << "start count time" << endl;
//    clock_t begin_t = clock();
    /* use 1st method to get the rough match result */
#if ROUGH_MATCH == 1
    results_first = filter->filter(query_descriptor);
#elif ROUGH_MATCH == 2
    
    /* use clustering method to get the rough match result */
    results_first = filter->filterCl(query_descriptor, 6);
    
    
#endif
//    cout << float(clock() - begin_t) / CLOCKS_PER_SEC << endl;
    
    
    delete filter;
    
    std::cout << "reduce finished" << std::endl;
    
    return 0;
    
    ////////////////////////////////////////////////////////// first step finish
    
    ////////////////////////////////////////////////////////// second step start
    int64 t = getTickCount();
    
    vector<s_templMatchGroupInfo> tmatchGInfos;
    
    vector<Mat> split_images;
    
    int depatch_col = 2;
    int depatch_row = 2;
    
    templateMatch* tMatch = new templateMatch;
    
    /* depatch one image into row*col subimages */
    tMatch->DepatchImage(query_img, split_images, depatch_row, depatch_col);
    
    for (auto group_idx = 0; group_idx < results_first.size(); ++group_idx) {
        s_templMatchGroupInfo tmGinfo;
        tmGinfo.finalSum = 0;
        
        /* get the group num */
        int groupNum = results_first[group_idx];
        tmGinfo.groupNum = groupNum;
        
        for (auto img_idx = 0; img_idx < all_images[groupNum].size(); ++img_idx) {
            s_templMatchImginfo tmIinfo;
            /* get the image */
            Mat img;
//            sprintf(filename, "%d/q_pic%d.jpg", groupNum + 1, img_idx + 1);
            img = imread(all_images[groupNum][img_idx], IMREAD_GRAYSCALE);
            tmIinfo.img_name = all_images[groupNum][img_idx];
            
            double tmpSum = 0;
            for (auto split_idx = 0; split_idx < depatch_row * depatch_col; ++ split_idx) {
                double minVal, maxVal;
                Mat img_disply;
                /* do the template matching
                *  "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED" 
                */
                tMatch->MatchingMethod(img, split_images[split_idx], CV_TM_SQDIFF_NORMED, minVal, maxVal, img_disply);
                tmpSum += tmpSum + minVal + maxVal;
                
                tmIinfo.minVal.push_back(minVal);
                tmIinfo.maxVal.push_back(maxVal);
                tmIinfo.imgDisp.push_back(img_disply);
            }
            tmIinfo.sumVal = tmpSum;
            
            tmGinfo.imgInfo.push_back(tmIinfo);
        }
        /* count the sum of the min 2 */
        sort(tmGinfo.imgInfo.begin(), tmGinfo.imgInfo.end(), structImgComparisonLess);
        
        tmGinfo.finalSum += tmGinfo.imgInfo[0].sumVal;
//        tmGinfo.finalSum += tmGinfo.imgInfo[1].sumVal;
        
        tmatchGInfos.push_back(tmGinfo);
    }
    
    ofstream pointFile;
    pointFile.open("routePoints.txt", ios_base::app);
#ifdef DEBUG_LEVEL_1
    /* debug only */
    for (auto i = 0; i < tmatchGInfos.size(); ++i) {
        for (auto j = 0; j < tmatchGInfos[i].imgInfo.size(); ++j) {
            cout << "the sum of image " << tmatchGInfos[i].imgInfo[j].img_name << " is: " << tmatchGInfos[i].imgInfo[j].sumVal << endl;
        }
    }
    
    string s_filename = fileInput.substr(fileInput.rfind("/") + 1);
    
    pointFile << s_filename << " ";
    
    for (auto i = 0; i < tmatchGInfos.size(); ++i) {
        cout << "the sum of group: " << tmatchGInfos[i].groupNum << " is: " << tmatchGInfos[i].finalSum << endl;
        pointFile << tmatchGInfos[i].groupNum << " ";
    }
#endif
    
    /* get the final group id */
    sort(tmatchGInfos.begin(), tmatchGInfos.end(), structGroupComparisonLess);
    int selected_group = tmatchGInfos[0].groupNum;
    
    pointFile << selected_group << "\n";

//    pointFile.close();
    
    cout << "the select group num is: " << selected_group <<  endl;
//    selected_group = 7;
    cout << "Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec";
//    return 0;

//    if (selected_group == 0 || selected_group == 5) {
//        cout << "point 0 and 5 is not ready..." << endl;
//    }
    ////////////////////////////////////////////////////////// second step end
    
    ////////////////////////////////////////////////////////// third step start
#if (G_INITIAL==0)
    initial = false;
#else
    initial = true;
#endif
//    initial =true;
    
    bool status = true;
    
    if (initial) {
        
        for (auto group_idx = 1; group_idx < numScenes; ++group_idx) {
            if(group_idx != 7) continue;
            frameStitch* fStitch = new frameStitch;
            
            vector<String> img_names;
            
            for (auto img_idx = 0; img_idx < all_images[group_idx].size(); ++img_idx) {
//                sprintf(filename, "%d/q_pic%d.jpg", group_idx + 1, img_idx + 1);
//                cout << fileLoc + filename << endl;
                img_names.push_back(all_images[group_idx][img_idx]);
            }
            
            fStitch->getImgNames(img_names);
            
            status = fStitch->resizeImages();
            
            if(!status){
                cout << "read and resize images failed..." << endl;
                return -1;
            }
            
            fStitch->findFeatures();
            
            fStitch->pairwiseMatch();
            
            fStitch->computeCameras(group_idx);
            
            fStitch->waveCorrection(group_idx);
            
            status = fStitch->warpImages();
            if(!status){
                cout << "warp images failed..." << endl;
                return -1;
            }
            
            fStitch->findSeamMask();
            
            fStitch->blendImages(group_idx, initial, fileInput);
            
            img_names.clear();
            delete fStitch;
        }
        
        cout << "initialization complete" << endl;
        
        return 0;
    }

    /* read the features of the frame */
//    Mat query_img_cl = imread(fileInput, IMREAD_GRAYSCALE);
    frameStitch* fStitch = new frameStitch;
    
    { //this part need to be replaced by optimal codes
        vector<String> img_names;
        
        for (auto i = 0; i < all_images[selected_group].size(); ++i) {
//            sprintf(filename, "%d/q_pic%d.jpg", selected_group + 1, i + 1);
            cout << all_images[selected_group][i] << endl;
            img_names.push_back(all_images[selected_group][i]);
        }
        
        fStitch->getImgNames(img_names);
        
        status = fStitch->resizeImages();
        
        fStitch->findFeatures();
        
    } //this part need to be replaced by optimal codes

    fStitch->findFrameFeature(fileInput);
    
    fStitch->pairwiseFrameMatch(fileInput);
    
    fStitch->computeFrameCamera();
    
//    fStitch->readCamerasOri(selected_group);
    
    fStitch->readCameras(selected_group);
    
    fStitch->computeFrameCameraPos();
    
//    fStitch->computeFrameCameraPosV2();
    fStitch->getPairKeypoints();
    
//    t = getTickCount();

    fStitch->computeFrameBundleAdjustment();
    
//    cout << "BundleAdjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec";
    
//    return 0;


    fStitch->waveCorrectionFrame();

    status = fStitch->warpImages();
    if(!status){
        cout << "warp images failed..." << endl;
        return -1;
    }
    
    fStitch->findSeamMask();
    
    fStitch->blendImages(selected_group, initial, fileInput);
    
    
}

