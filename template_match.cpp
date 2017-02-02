//
//  template_match.cpp
//  opencv
//
//  Created by yu zhang on 16/6/12.
//  Copyright © 2016年 yu zhang. All rights reserved.
//

#include "template_match.hpp"

bool structImgComparisonLess(const s_templMatchImginfo &a,const s_templMatchImginfo &b)
{
    return a.sumVal < b.sumVal;
}

bool structGroupComparisonLess(const s_templMatchGroupInfo &a,const s_templMatchGroupInfo &b){
    return a.finalSum < b.finalSum;
}



void templateMatch::MatchingMethod(Mat img, Mat templ, int match_method, double& minValRaw, double& maxValRaw, Mat& img_display )
{
    Mat result;
    
    /// Source image to display
    img.copyTo( img_display );
    
    /// Create the result matrix
    int result_cols =  img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    
    result.create(result_rows, result_cols, CV_32FC1);
    
    /// Do the Matching and Normalize
    matchTemplate( img, templ, result, match_method );
    //    normalize( img_result, img_result, 0, 1, NORM_MINMAX, -1, Mat() );
    
    /// Localizing the best match with minMaxLoc
    Point minLocRaw; Point maxLocRaw;
    Point matchLoc;
    
    //    minMaxLoc( img_result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
    minMaxLoc( result, &minValRaw, &maxValRaw, &minLocRaw, &maxLocRaw, Mat() );
    
    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLocRaw; }
    else
    { matchLoc = maxLocRaw; }
    
    /// Show me what you got
    rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
    
    //    cout << "minVal= " << minValRaw << " maxval= " << maxValRaw << endl;
    
}

// dipatch image to serveral parts
void templateMatch::DepatchImage(Mat src_image, vector<Mat>& output, int num, int cell){
    /* get the origin image size */
    Size s = src_image.size();
    /* compute each sub image size */
    float sub_width = s.width / num;
    float sub_height = s.height / cell;
    
    for (int j = 0; j < cell; ++j) {
        for (int i = 0; i < num; ++i) {
            output.push_back(src_image(Rect(i * sub_width, j * sub_height, sub_width, sub_height)));
        }
    }
    return;
}