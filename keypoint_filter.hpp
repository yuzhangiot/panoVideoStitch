//
//  keypoint_filter.hpp
//  PhotoTourism
//
//  Created by Bichuan Guo on 6/1/16.
//  Copyright © 2016 loft. All rights reserved.
//

#ifndef keypoint_filter_hpp
#define keypoint_filter_hpp

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/stitching/detail/matchers.hpp"
#include "FileStorageExt.h"
#include <time.h>


class KeyPointFilter {
public:
    void create(int scenes);
    void createFull(int scenes);
    void addTrainSet(cv::Mat descriptor, int sceneIdx);
    cv::detail::ImageFeatures addTrainSetFull(cv::Mat img, int imgIdx, int imgNum);
    void reduce();
    void reduceVecVec(std::vector<std::vector<cv::Mat>> input, std::vector<cv::Mat>& output, int thresh);
    void reduceMat(cv::Mat input1, cv::Mat input2, cv::Mat& output, int thresh);
    std::vector<int> filter(cv::Mat query);
    std::vector<int> filterCl(cv::Mat query, int method);
    void clfileToYml(int method);
    void loadcl(int method);
    void save();
    void load();
    void saveCommon(std::vector<cv::Mat> inputMatV, string filename);
    void loadCommon(std::vector<cv::Mat>& outputMatV, string filename, int clnum);
    void clustering();
    
protected:
    std::vector<std::vector<cv::Mat> > m_trainSet;
public:
    std::vector<cv::Mat> m_dict;
    std::vector<cv::Mat> allClusterRe;

    std::vector<std::pair<float, int> > m_metrics1, m_metrics2, m_metrics3;
};

#endif /* keypoint_filter_hpp */
