//
//  template_match.hpp
//  opencv
//
//  Created by yu zhang on 16/6/12.
//  Copyright © 2016年 yu zhang. All rights reserved.
//

#ifndef template_match_hpp
#define template_match_hpp

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct CV_EXPORTS s_templMatchImginfo{
    string img_name;
    vector<double> minVal;
    vector<double> maxVal;
    vector<Mat> imgDisp;
    double sumVal;
};

struct CV_EXPORTS s_templMatchGroupInfo{
    int groupNum;
    vector<s_templMatchImginfo> imgInfo;
    double finalSum; //equie to the sum of the minest 2
};

bool structImgComparisonLess(const s_templMatchImginfo &a,const s_templMatchImginfo &b);

bool structGroupComparisonLess(const s_templMatchGroupInfo &a,const s_templMatchGroupInfo &b);

class templateMatch {
  
public:
    void MatchingMethod(Mat , Mat , int , double& , double& , Mat& );
    void DepatchImage(Mat, vector<Mat>&, int, int);
    
protected:

    
};


#endif /* template_match_hpp */

