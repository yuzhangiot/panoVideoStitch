/*
* Copyright (c) Statsmaster Ltd, all rights reserved
*
* This is the collection of extra functions for OpenCV's FileStorage.
*/

#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/camera.hpp"

using std::vector;
using std::string;
using cv::RotatedRect;
using cv::FileStorage;
using cv::Mat;

// FileStorage functions for save vector<...> (... have ambiguous size)
template <typename T>
FileStorage & operator << (FileStorage & fs, const vector<vector<T>> & vecs);

// FileStorage functions for read vector<...> (... have ambiguous size)
template <typename T>
void operator >> (const cv::FileNode& node, vector<vector<T>>& vecs);

// FileStorage functions to write RotatedRect
FileStorage & operator << (FileStorage & fs, const RotatedRect & ell);
FileStorage & operator << (FileStorage & fs, const vector<RotatedRect> & ellipses);

// FileStorage functions to read RotatedRect
void operator >> (const cv::FileNode& node, RotatedRect & vec);
void operator >> (const cv::FileNode& node, vector<RotatedRect> & vecs);

// FileStorage functions to write/read vector<Mat>
FileStorage& operator << (FileStorage& fs, const vector<Mat> & vecs);
void operator >> (const cv::FileNode& node, vector<Mat> & vecs);

// FileStorage functions to write/read vector<string>
FileStorage& operator << (FileStorage& fs, const vector<string> & vecs);
void operator >> (const cv::FileNode& node, vector<string> & vecs);

// FileStorage functions to write/read RotatedRect
FileStorage & operator << (FileStorage & fs, const cv::PCA & pca);
void operator >> (const cv::FileNode& node, cv::PCA & pca);

// FileStorage functions to write/read MatchInfo
FileStorage & operator << (FileStorage & fs, const cv::detail::MatchesInfo & m_info);
void operator >> (const cv::FileNode& node, cv::detail::MatchesInfo & m_info);

// FileStorage functions to write/read vector<cv::detail::MatchesInfo>
FileStorage& operator << (FileStorage& fs, const vector<cv::detail::MatchesInfo> & vecs);
void operator >> (const cv::FileNode& node, vector<cv::detail::MatchesInfo> & vecs);

// FileStorage functions to write/read CameraParams
FileStorage & operator << (FileStorage & fs, const cv::detail::CameraParams & cam);
void operator >> (const cv::FileNode& node, cv::detail::CameraParams & cam);

// FileStorage functions to write/read vector<cv::detail::MatchesInfo>
FileStorage& operator << (FileStorage& fs, const vector<cv::detail::CameraParams> & vecs);
void operator >> (const cv::FileNode& node, vector<cv::detail::CameraParams> & vecs);




