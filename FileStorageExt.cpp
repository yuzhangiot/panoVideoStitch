/*
* Copyright (c) Statsmaster Ltd, all rights reserved
*
* This is the collection of extra functions for OpenCV's FileStorage.
*/

#include "FileStorageExt.h"

using namespace std;
using namespace cv;
using cv::detail::MatchesInfo;
using cv::detail::CameraParams;
using cv::detail::ImageFeatures;

// FileStorage functions for vector<vector<int>>
template <typename T>
FileStorage & operator << (FileStorage & fs, const vector<vector<T>> & vecs)
{
    fs << "{" << "number" << (int)vecs.size();

    for (size_t i = 0; i < vecs.size(); i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        fs << idx.str() << vecs[i];
    }

    fs << "}";
    return fs;
}
template FileStorage & operator << (FileStorage & fs, const vector<vector<int>> & vecs);
template FileStorage & operator << (FileStorage & fs, const vector<vector<float>> & vecs);
template FileStorage & operator << (FileStorage & fs, const vector<vector<Point>> & vecs);
template FileStorage & operator << (FileStorage & fs, const vector<vector<Point2f>> & vecs);
template FileStorage & operator << (FileStorage & fs, const vector<vector<MatchesInfo>> & vecs);

// FileStorage functions for read vector<...> (... have ambiguous size)
template <typename T>
void operator >> (const FileNode& node, vector<vector<T>>& vecs)
{
    int num = node["number"];
    vecs.resize(num);
    for (size_t i = 0; i < num; i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        node[idx.str()] >> vecs[i];
    }
}
template void operator >> (const cv::FileNode& node, vector<vector<int>>& vecs);
template void operator >> (const cv::FileNode& node, vector<vector<float>>& vecs);
template void operator >> (const cv::FileNode& node, vector<vector<Point>>& vecs);
template void operator >> (const cv::FileNode& node, vector<vector<Point2f>>& vecs);
template void operator >> (const cv::FileNode& node, vector<vector<MatchesInfo>>& vecs);

// FileStorage function to write RotatedRect
FileStorage& operator << (FileStorage& fs, const RotatedRect & ell)
{
    fs << "{" << "x" << ell.center.x << "y" << ell.center.y;
    fs << "angle" << ell.angle << "width" << ell.size.width << "height" << ell.size.height << "}";
    return fs;
}

// FileStorage functions to read RotatedRect
void operator >> (const FileNode& node, RotatedRect & item)
{
    node["x"] >> item.center.x;
    node["y"] >> item.center.y;
    node["angle"] >> item.angle;
    node["width"] >> item.size.width;
    node["height"] >> item.size.height;
}

FileStorage& operator << (FileStorage& fs, const vector<RotatedRect> & ellipses)
{
    fs << "{" << "number" << (int)ellipses.size();

    for (size_t i = 0; i < ellipses.size(); i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        fs << idx.str() << ellipses[i];
    }

    fs << "}";
    return fs;
}

void operator >> (const FileNode& node, vector<RotatedRect> & vecs)
{
    int num = node["number"];
    vecs.resize(num);
    for (size_t i = 0; i < num; i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        node[idx.str()] >> vecs[i];
    }
}

// FileStorage functions to write vector<Mat>
FileStorage& operator << (FileStorage& fs, const vector<Mat> & vecs)
{
    fs << "{" << "number" << (int)vecs.size();

    for (size_t i = 0; i < vecs.size(); i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        fs << idx.str() << vecs[i];
    }

    fs << "}";
    return fs;
}

// FileStorage functions to read vector<Mat>
void operator >> (const FileNode& node, vector<Mat> & vecs)
{
    int num = node["number"];
    vecs.resize(num);
    for (size_t i = 0; i < num; i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        node[idx.str()] >> vecs[i];
    }
}

// FileStorage functions to write vector<string>
FileStorage& operator << (FileStorage& fs, const vector<string> & matrixes)
{
    fs << "{" << "number" << (int)matrixes.size();

    for (size_t i = 0; i < matrixes.size(); i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        fs << idx.str() << matrixes[i];
    }

    fs << "}";
    return fs;
}

// FileStorage functions to read vector<string>
void operator >> (const FileNode& node, vector<string> & vecs)
{
    int num = node["number"];
    vecs.resize(num);
    for (size_t i = 0; i < num; i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        node[idx.str()] >> vecs[i];
    }
}

// FileStorage functions to write/read RotatedRect
FileStorage & operator << (FileStorage & fs, const PCA & pca)
{
    fs << "{";
    fs << "mean" << pca.mean;
    fs << "eigenvalues" << pca.eigenvalues;
    fs << "eigenvectors" << pca.eigenvectors;
    fs << "}";
    return fs;
}

void operator >> (const FileNode& node, PCA & pca)
{
    node["mean"] >> pca.mean;
    node["eigenvalues"] >> pca.eigenvalues;
    node["eigenvectors"] >> pca.eigenvectors;
}

// FileStorage functions to write/read MatchInfo
FileStorage & operator << (FileStorage & fs, const cv::detail::MatchesInfo & m_info)
{
    fs << "{";
    fs << "src_img_idx" << m_info.src_img_idx;
    fs << "dst_img_idx" << m_info.dst_img_idx;
    fs << "matches" << m_info.matches;
    fs << "inliers_mask" << m_info.inliers_mask;
    fs << "num_inliers" << m_info.num_inliers;
    fs << "H" << m_info.H;
    fs << "confidence" << m_info.confidence;
    fs << "}";
    return fs;
}

void operator >> (const cv::FileNode& node, cv::detail::MatchesInfo & m_info)
{
    node["src_img_idx"] >> m_info.src_img_idx;
    node["dst_img_idx"] >> m_info.dst_img_idx;
    node["matches"] >> m_info.matches;
    node["inliers_mask"] >> m_info.inliers_mask;
    node["num_inliers"] >> m_info.num_inliers;
    node["H"] >> m_info.H;
    node["confidence"] >> m_info.confidence;
}

// FileStorage functions to write vector<MatchesInfo>
FileStorage& operator << (FileStorage& fs, const vector<MatchesInfo> & vecs)
{
    fs << "{" << "number" << (int)vecs.size();

    for (size_t i = 0; i < vecs.size(); i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        fs << idx.str() << vecs[i];
    }

    fs << "}";
    return fs;
}

// FileStorage functions to read vector<MatchesInfo>
void operator >> (const FileNode& node, vector<MatchesInfo> & vecs)
{
    int num = node["number"];
    vecs.resize(num);
    for (size_t i = 0; i < num; i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        node[idx.str()] >> vecs[i];
    }
}

// FileStorage functions to write/read MatchInfo
FileStorage & operator << (FileStorage & fs, const cv::detail::CameraParams & cam)
{
    fs << "{";
    fs << "focal" << cam.focal;
    fs << "aspect" << cam.aspect;
    fs << "ppx" << cam.ppx;
    fs << "ppy" << cam.ppy;
    fs << "R" << cam.R;
    fs << "t" << cam.t;
    fs << "}";
    return fs;
}

void operator >> (const cv::FileNode& node, cv::detail::CameraParams & cam)
{
    node["focal"] >> cam.focal;
    node["aspect"] >> cam.aspect;
    node["ppx"] >> cam.ppx;
    node["ppy"] >> cam.ppy;
    node["R"] >> cam.R;
    node["t"] >> cam.t;
}

// FileStorage functions to write vector<CameraParams>
FileStorage& operator << (FileStorage& fs, const vector<CameraParams> & vecs)
{
    fs << "{" << "number" << (int)vecs.size();

    for (size_t i = 0; i < vecs.size(); i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        fs << idx.str() << vecs[i];
    }

    fs << "}";
    return fs;
}

// FileStorage functions to read vector<CameraParams>
void operator >> (const FileNode& node, vector<CameraParams> & vecs)
{
    int num = node["number"];
    vecs.resize(num);
    for (size_t i = 0; i < num; i++)
    {
        ostringstream idx;
        idx << "idx-" << i;
        node[idx.str()] >> vecs[i];
    }
}










