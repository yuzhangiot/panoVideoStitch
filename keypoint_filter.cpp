//
//  keypoint_filter.cpp
//  PhotoTourism
//
//  Created by Bichuan Guo on 6/1/16.
//  Copyright © 2016 loft. All rights reserved.
//

#include "keypoint_filter.hpp"

using namespace std;
using namespace cv;

bool pairComparisonLess(const pair<int,double> &a,const pair<int,double> &b)
{
    return a.second < b.second;
}

bool pairComparisonMore(const pair<int,double> &a,const pair<int,double> &b)
{
    return a.second > b.second;
}

void KeyPointFilter::create(int scenes) {
    m_trainSet.clear();
    for (int i = 0; i < scenes; i++) {
        m_trainSet.push_back(std::vector<cv::Mat>());
    }
}

void KeyPointFilter::createFull(int scenes){
//    m_trainSetFull.clear();
}

void KeyPointFilter::addTrainSet(cv::Mat descriptor, int sceneIdx) {
    if (sceneIdx >= m_trainSet.size())
        return;
    m_trainSet[sceneIdx].push_back(descriptor);
}

cv::detail::ImageFeatures KeyPointFilter::addTrainSetFull(cv::Mat img, int imgIdx, int imgNum){
    
    Ptr<cv::detail::FeaturesFinder> finder;
    finder = makePtr<cv::detail::SurfFeaturesFinder>();
    
    cv::detail::ImageFeatures local_features;
    
    /* init features */
    (*finder)(img, local_features);
    local_features.img_idx = imgIdx;
    
    finder->collectGarbage();
    
    return local_features;
}

void KeyPointFilter::reduce() {
    
    double thresh = 300.f;
    
    m_dict.clear();
    
    for (int sceneIdx = 0; sceneIdx < m_trainSet.size(); sceneIdx++) {
        std::vector<std::vector<cv::Mat> > bucket;
        std::vector<cv::Mat> avg;
        bucket.clear();
        avg.clear();
        
        std::cout << "scene " << sceneIdx << std::endl;
        
        for (int i = 0; i < m_trainSet[sceneIdx].size(); i++) {
            
            std::cout << "  descriptor " << i << std::endl;
            
            for (int rowIdx = 0; rowIdx < m_trainSet[sceneIdx][i].rows; rowIdx++) {
                
                //std::cout << "    row " << rowIdx << std::endl;
                
                cv::Mat rowVector = m_trainSet[sceneIdx][i].row(rowIdx);
                bool match = false;
                for (int optionIdx = 0; optionIdx < bucket.size(); optionIdx++) {
                    double dist = cv::norm(rowVector, avg[optionIdx]);
                    if (dist < thresh) {
                        //std::cout << "      matched option " << optionIdx << std::endl;
                        match = true;
                        bucket[optionIdx].push_back(rowVector);
                        cv::Mat avgMat = cv::Mat::zeros(rowVector.size(), rowVector.type());
                        for (int j = 0; j < bucket[optionIdx].size(); j++) {
                            avgMat = avgMat + bucket[optionIdx][j];
                        }
                        avgMat = avgMat / bucket[optionIdx].size();
                        avg[optionIdx] = avgMat;
                    }
                }
                if (!match) {
                    //std::cout << "      new option " << bucket.size() << std::endl;
                    std::vector<cv::Mat> newslot(1);
                    newslot[0] = rowVector;
                    bucket.push_back(newslot);
                    avg.push_back(rowVector);
                }
            }
        }
        
        cv::Mat word;
        word.create((int)avg.size(), avg[0].cols, avg[0].type());
        for (int i = 0; i < avg.size(); i++) {
            avg[i].row(0).copyTo(word.row(i));
        }
        m_dict.push_back(word);
    }
    
    cout << "m_dict's size = " << m_dict.size() << endl;
    cout << "m_dict's row = " << m_dict[0].rows << endl;
    
    return;
}

void KeyPointFilter::reduceVecVec(vector<vector<cv::Mat>> inputMatVV, vector<cv::Mat>& outputMatV, int thresh){
    for (int sceneIdx = 0; sceneIdx < inputMatVV.size(); ++sceneIdx) {
//        if(sceneIdx != 1) continue;
        
        
        cv::Mat reMat(inputMatVV[sceneIdx][0].rows, inputMatVV[sceneIdx][0].cols, inputMatVV[sceneIdx][0].type());

        
        cout << "cluster " << sceneIdx << endl;
        
        for (auto ptIdx = 0; ptIdx < inputMatVV[sceneIdx].size(); ++ptIdx) {
            cout << "point " << ptIdx << endl;
            
            // copy the first point to reMat
            if (ptIdx == 0) {
                inputMatVV[sceneIdx][ptIdx].copyTo(reMat);
            }
            else{
                reduceMat(inputMatVV[sceneIdx][ptIdx], reMat, reMat, thresh);
            }
        }
        
        outputMatV.push_back(reMat);
        
    }
}

void KeyPointFilter::reduceMat(cv::Mat input1Mat, cv::Mat input2Mat, cv::Mat &outputMat, int thresh){
    std::vector<int> totalNum;
    std::vector<cv::Mat> avg;
    std::vector<cv::Mat> newavg;
    avg.clear();
    newavg.clear();
    
    // first put all vectors of mat2 into bucket and avg
    for (auto row2Idx = 0; row2Idx < input2Mat.rows; ++row2Idx) {
        avg.push_back(input2Mat.row(row2Idx));
        totalNum.push_back(1);
    }
    
    // Then compare all vectors of mat1 with mat2
    for (auto rowIdx = 0; rowIdx < input1Mat.rows; ++rowIdx) {
//        cout << "row " << rowIdx << endl;
        
        cv::Mat rowVector = input1Mat.row(rowIdx);
        bool match = false;
        for (auto optionIdx = 0; optionIdx < avg.size(); ++optionIdx) {
            
//            int64 t = getTickCount();
            
            double dist = cv::norm(rowVector, avg[optionIdx]);
            
//            cout << "norm, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

            if (dist < thresh) {
//                t = getTickCount();
                match = true;
//                bucket[optionIdx].push_back(rowVector);
                cv::Mat avgMat = cv::Mat::zeros(rowVector.size(), rowVector.type());
//                for (auto j = 0; j < bucket[optionIdx].size(); ++j) {
                avgMat = avgMat + avg[optionIdx] * totalNum[optionIdx] + rowVector;
//                }
                totalNum[optionIdx] += 1;
                avgMat = avgMat / totalNum[optionIdx];
                avg[optionIdx] = avgMat.clone();
                
//                cout << "count avg, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
            }
        }
        if (!match) {
            newavg.push_back(rowVector);
        }
    }
    
    avg.insert(avg.end(), newavg.begin(), newavg.end());
    
    cv::Mat word;
    word.create((int)avg.size(), avg[0].cols, avg[0].type());
    for (auto i = 0; i < avg.size(); ++i) {
        avg[i].row(0).copyTo(word.row(i));
    }
    
    outputMat.release();
    word.copyTo(outputMat);
}

void KeyPointFilter::clustering(){
    cout << "clustering function" << endl;
    
    cv::FlannBasedMatcher matcher;
    vector<vector<int>> pair_result;
    pair_result.clear();
    
    int goodMatchValue = 100;
    
    std::vector<cv::DMatch> pair_matches;
    
    for (auto rowIdx = 0; rowIdx < m_dict.size(); ++rowIdx) {
        vector<int> m_line;
        m_line.clear();
        for (auto colIdx = 0; colIdx < m_dict.size(); ++colIdx) {
            if(colIdx == rowIdx)
                continue;
            int numGoodMatch = 0;
            matcher.match(m_dict[rowIdx], m_dict[colIdx], pair_matches);
            
            //pick the bigger row num
//            int rowNum = (m_dict[rowIdx].rows >= m_dict[colIdx].rows) ? m_dict[rowIdx].rows : m_dict[colIdx].rows;
            for (auto i = 0; i < m_dict[rowIdx].rows; ++i) {
                if (pair_matches[i].distance < goodMatchValue) {
                    numGoodMatch++;
                }
            }
            
            m_line.push_back(numGoodMatch);
            
            cout << "distance from " << rowIdx << " to " << colIdx << " is: " << numGoodMatch << endl;
        }
        pair_result.push_back(m_line);
        
        cout << endl;
    }
    
    /* write the value of the pair */
    FileStorage fs("loc_pairs.yml", FileStorage::WRITE);
    fs << "pairs" << pair_result;
    fs.release();
    
    /* read the value of the pair */
    vector<vector<int>> pairs;
    FileStorage fs_r("loc_pairs.yml", FileStorage::WRITE);
    fs_r["pairs"] >> pairs;
    fs_r.release();

}


std::vector<int> KeyPointFilter::filter(cv::Mat query) {
    
//    std::vector<int> options;
    
    cv::FlannBasedMatcher matcher;
    
    /* allocate arrays for mid-result */
    int pick_num = 1;
    vector<pair<int, double>> min_dist_results;
    vector<pair<int, int>> num_goodMatch_results;
    vector<pair<int, double>> avg_goodMatch_results;
    vector<int> final_results;
    
    
    for (int sceneIdx = 0; sceneIdx < m_dict.size(); sceneIdx++) {
        std::vector<cv::DMatch> matches;
        if (sceneIdx == 7) {
            cout << "cluster 7 size is: " << m_dict[sceneIdx].rows << endl;
            cout << "descriptor's dimension is: " << m_dict[sceneIdx].cols << endl;
        }
        matcher.match(query, m_dict[sceneIdx], matches);
        
        double max_dist = 0; double min_dist = 100;
        for( int i = 0; i < query.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        printf("Scene %d:\n", sceneIdx + 1);
        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );
        
        int numGoodMatches = 0;
        double GoodMatchesDistance = 0.0f;
        
        for (int i = 0; i < query.rows; i++) {
            if (matches[i].distance <= 100.f) {
                numGoodMatches++;
                GoodMatchesDistance += matches[i].distance;
            }
        }
        printf("-- Good matches : %d %lf \n", numGoodMatches, GoodMatchesDistance);
        
        /* find the best x matches for each param */
        double avgGoodMatches = GoodMatchesDistance / numGoodMatches;
        
        // min_dist
        if (min_dist_results.size() < pick_num) {
            pair<int, double> tmpPair = {sceneIdx, min_dist};
            min_dist_results.push_back(tmpPair);
            /* sort the min dist array */
            sort(min_dist_results.begin(), min_dist_results.end(), pairComparisonMore);
        }
        else{
            if (min_dist_results[0].second > min_dist) {
                min_dist_results[0].first = sceneIdx;
                min_dist_results[0].second = min_dist;
                sort(min_dist_results.begin(), min_dist_results.end(), pairComparisonMore);
            }
        }
        
        // number of good matches
        if (num_goodMatch_results.size() < pick_num) {
            num_goodMatch_results.push_back({sceneIdx, numGoodMatches});
            sort(num_goodMatch_results.begin(), num_goodMatch_results.end(), pairComparisonLess);
        }
        else{
            if (num_goodMatch_results[0].second < numGoodMatches) {
                num_goodMatch_results[0].first = sceneIdx;
                num_goodMatch_results[0].second = numGoodMatches;
                sort(num_goodMatch_results.begin(), num_goodMatch_results.end(), pairComparisonLess);
            }
        }
        
        //average distance of good matches
        if (avg_goodMatch_results.size() < pick_num) {
            avg_goodMatch_results.push_back({sceneIdx, avgGoodMatches});
            sort(avg_goodMatch_results.begin(), avg_goodMatch_results.end(), pairComparisonMore);
        }
        else{
            if (avg_goodMatch_results[0].second > avgGoodMatches) {
                cout << "now the avg good match index is: " << sceneIdx << ", the value is: " << avgGoodMatches << endl;
                avg_goodMatch_results[0].first = sceneIdx;
                avg_goodMatch_results[0].second = avgGoodMatches;
                sort(avg_goodMatch_results.begin(), avg_goodMatch_results.end(), pairComparisonMore);
            }
        }
        
    }
    
    /* Then de-duplication */
    for (auto i = 0; i < pick_num; ++i) {
        final_results.push_back(min_dist_results[i].first);
        final_results.push_back(num_goodMatch_results[i].first);
        final_results.push_back(avg_goodMatch_results[i].first);
    }
    sort(final_results.begin(), final_results.end());
    final_results.erase(unique(final_results.begin(), final_results.end()), final_results.end());
    
//    ofstream tmpfile;
//    tmpfile.open("tmpfile.txt", ios_base::app);
    // debug only
    for (auto i = final_results.begin(); i != final_results.end(); ++i) {
        cout << *i << endl;
//        tmpfile << *i << "\n";
    }
    
    
    return final_results;
}

void KeyPointFilter::loadcl(int method){
    string fileName = "";
    int clnum = 0;
    
    /* read clustering result from diff method */
    if (method == 1) {
        fileName = "saveClusterAp1_250.yml";
        clnum = 9;
        loadCommon(allClusterRe, fileName.c_str(), clnum);
    }
    else if (method == 2){
        fileName = "saveClusterAp2.yml";
        clnum = 12;
        loadCommon(allClusterRe, fileName.c_str(), clnum);
    }
    else if (method == 3){
        fileName = "saveClusterAp3.yml";
        clnum = 6;
        loadCommon(allClusterRe, fileName.c_str(), clnum);
    }
    else if (method == 4){
        fileName = "saveClusterAp4.yml";
        clnum = 5;
        loadCommon(allClusterRe, fileName.c_str(), clnum);
    }
    else if (method == 5){
        fileName = "saveClusterAp5.yml";
        clnum = 5;
        loadCommon(allClusterRe, fileName.c_str(), clnum);
    }
    else if (method == 6){
        fileName = "saveClusterAp6.yml";
        clnum = 10;
        loadCommon(allClusterRe, fileName.c_str(), clnum);
    }
    else{
        cerr << "unsupported method type!" << endl;
        terminate();
    }

}

vector<int> KeyPointFilter::filterCl(cv::Mat query, int method){
    cout << "start count time" << endl;
    clock_t begin_t = clock();
    
    vector<int> final_result;
    int pickNum = 2;


    // define a fast lib ANN matcher
    vector<pair<int, int>> num_goodmatch_cl;
    vector<pair<int, int>> num_goodmatch_pt;
    cv::FlannBasedMatcher matcher;
    int euthresh = 250;
    
    for (auto clIdx = 0; clIdx < allClusterRe.size(); ++clIdx) {
        if (clIdx == 2) {
            cout << "the cluter's rows: " << allClusterRe[clIdx].rows << endl;
        }
        vector<cv::DMatch> matches;
        matcher.match(query, allClusterRe[clIdx], matches);
        
        int numGoodMatches = 0;
        
        for (auto rowIdx = 0; rowIdx < query.rows; ++rowIdx) {
            if (matches[rowIdx].distance <= euthresh) {
                ++numGoodMatches;
            }
        }
        
        pair<int, int> tmppair = {clIdx, numGoodMatches};
        num_goodmatch_cl.push_back(tmppair);
        
//        cout << "cluster " << clIdx << endl;
//        cout << "good matches " << numGoodMatches << endl;
    }
    
    sort(num_goodmatch_cl.begin(), num_goodmatch_cl.end(), pairComparisonMore);
    
    
    cout << "The best fit cluster is: " << num_goodmatch_cl[0].first << endl;
    
    /////// read cluster info
    ifstream apfile;
    string s_line;
    vector<vector<string>> alltokens;
    string apfileName = "ap" + to_string(method) + ".txt";
    
    apfile.open(apfileName.c_str());
    if(apfile.is_open()){
        while (getline(apfile, s_line)) {
            istringstream iss(s_line);
            vector<string> tmptokens{std::istream_iterator<string>{iss}, std::istream_iterator<string>{}};
            alltokens.push_back(tmptokens);
        }
    }
    /////// end read cluster info
    
    int clIndic = num_goodmatch_cl[0].first;
    
    // if size <= 2, add to final result directly
    if (alltokens[clIndic].size() < pickNum + 1) {
        for (auto i = 0; i < alltokens[clIndic].size(); ++i) {
            final_result.push_back(stoi(alltokens[clIndic][i]) - 1);
        }
    }
    else{ // if size > 2, find 2 best points
        for (auto ptIdx = 0; ptIdx < alltokens[clIndic].size(); ++ptIdx) {
            vector<cv::DMatch> pt_matches;
            int ptIndic = stoi(alltokens[clIndic][ptIdx]) - 1;
            
            matcher.match(query, m_dict[ptIndic], pt_matches);
            
            int numGoodMatches = 0;
            
            for (auto rowIdx = 0; rowIdx < query.rows; ++rowIdx) {
                if (pt_matches[rowIdx].distance <= euthresh) {
                    ++numGoodMatches;
                }
            }
            
//            cout << "point is: " << ptIndic << endl;
//            cout << "good matches is: " << numGoodMatches << endl;
            
            num_goodmatch_pt.push_back({ptIndic, numGoodMatches});
        }
        
        sort(num_goodmatch_pt.begin(), num_goodmatch_pt.end(), pairComparisonMore);
        
        for (auto i = 0; i < pickNum; ++i) {
            final_result.push_back(num_goodmatch_pt[i].first);
        }
    }
    
    auto end_t = float(clock() - begin_t) / CLOCKS_PER_SEC;
    cout << end_t << endl;

    
    // write the result to file
    ofstream m_file, m_file_time;
    m_file.open("cl_match.txt", ios_base::app);
    m_file_time.open("cl_match_time.txt", ios_base::app);
    
    m_file << num_goodmatch_cl[0].first << "\n";
    m_file_time << end_t << "\n";
    
    for (auto i = 0; i < final_result.size(); ++i) {
        cout << final_result[i] << endl;
    }
    
    
    return final_result;
}


void KeyPointFilter::clfileToYml(int method){
    
    /*read diff txt according to diff method,
    * but this part should be replaced read txt to using diff clustering method
    */
    ifstream apfile;
    string s_line;
    vector<vector<string>> alltokens;
    string outfilename = "";
    if (method == 1) {
        apfile.open("ap1.txt");
        outfilename = "saveClusterAp1.yml";
    }
    else if (method == 2){
        apfile.open("ap2.txt");
        outfilename = "saveClusterAp2.yml";
    }
    else if (method == 3){
        apfile.open("ap3.txt");
        outfilename = "saveClusterAp3.yml";
    }
    else if (method == 4){
        apfile.open("ap4.txt");
        outfilename = "saveClusterAp4.yml";
    }
    else if (method == 5){
        apfile.open("ap5.txt");
        outfilename = "saveClusterAp5.yml";
    }
    else{
        cerr << "unsuppot method!" << endl;
        terminate();
    }
    
    /* read clusters */
    if(apfile.is_open()){
        while (getline(apfile, s_line)) {
            istringstream iss(s_line);
            vector<string> tmptokens{std::istream_iterator<string>{iss}, std::istream_iterator<string>{}};
            alltokens.push_back(tmptokens);
        }
    }
    
    //debug purpose
//    for (auto i = 0; i < alltokens.size(); ++i) {
//        for (auto j = 0; j < alltokens[i].size(); ++j) {
//            cout << alltokens[i][j] << " ";
//        }
//        cout << endl;
//    }
    
    cv::FlannBasedMatcher matcher;
    
    /* combine points for each cluster and then duplicate */
    // 1. combine points
//    vector<cv::Mat> allclusters(alltokens.size());
    vector<vector<cv::Mat>> allclusters;
    for (auto clusterIdx = 0; clusterIdx < alltokens.size(); ++clusterIdx) {
        vector<cv::Mat> tmpcluster;
        // iter points in each cluster
        for (auto pointIdx = 0; pointIdx < alltokens[clusterIdx].size(); ++pointIdx) {
            int pos = stoi(alltokens[clusterIdx][pointIdx]) - 1;
//            cout << "cluster: " << clusterIdx << "  point: " << pos << endl;
            tmpcluster.push_back(m_dict[pos]);
        }
//        cv::vconcat(tmpcluster, allclusters[clusterIdx]);
        allclusters.push_back(tmpcluster);
    }
    
    // debug purpose
    for (auto i = 0; i < m_dict.size(); ++i) {
        cout << "the " << i << " point has: " << m_dict[i].rows << " descriptors" << endl;
    }
    for (auto i = 0; i < allclusters.size(); ++i) {
        int tmpSum = 0;
        for (auto j = 0; j < allclusters[i].size(); ++j) {
            tmpSum += allclusters[i][j].rows;
        }
        cout << "the " << i << " cluster has: " << tmpSum <<" descrptors" << endl;
    }
    
    // 2. reduplicate each cluster
    vector<cv::Mat> allclusterRe;
    reduceVecVec(allclusters, allclusterRe, 300);
    
    //debug purpose
    for (auto i = 0; i < allclusterRe.size(); ++i) {
        cout << "the " << i << " cluster has: " << allclusterRe[i].rows <<" descrptors" << endl;
    }
    
    /* save reduced descriptors */
    saveCommon(allclusterRe, outfilename.c_str());
    
    
}


void KeyPointFilter::save() {
    cv::FileStorage fs("save.yml", cv::FileStorage::WRITE);
    for (int sceneIdx = 0; sceneIdx < m_dict.size(); sceneIdx++) {
        char sceneName[20];
        sprintf(sceneName, "scene%d", sceneIdx);
        fs << sceneName << m_dict[sceneIdx];
    }
    fs.release();
}

void KeyPointFilter::saveCommon(vector<cv::Mat> inputMatV, string filename){
    cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
    for (auto sceneIdx = 0; sceneIdx < inputMatV.size(); ++sceneIdx) {
        char sceneName[20];
        sprintf(sceneName, "scene%d", sceneIdx);
        fs << sceneName << inputMatV[sceneIdx];
    }
    fs.release();
}

void KeyPointFilter::loadCommon(std::vector<cv::Mat> &outputMatV, string filename, int clnum){
    cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
    cv::Mat tmp;
    for (auto i = 0; i < clnum; ++i) { //outputMatV has to be set size first
        char sceneName[20];
        sprintf(sceneName, "scene%d", i);
        fs[sceneName] >> tmp;
        outputMatV.push_back(tmp);
    }
    fs.release();
}

void KeyPointFilter::load() {
    cv::FileStorage fs("save.yml", cv::FileStorage::READ);
    cv::Mat tmp;
    for (int sceneIdx = 0; sceneIdx < m_trainSet.size(); sceneIdx++) {
        char sceneName[20];
        sprintf(sceneName, "scene%d", sceneIdx);
        fs[sceneName] >> tmp;
        m_dict.push_back(tmp);
    }
    fs.release();
    
    cout << "point 6's number is: " << m_dict[6].rows << endl;
    cout << "point 6's number is: " << m_dict[7].rows << endl;
    cout << "point 6's number is: " << m_dict[17].rows << endl;
    cout << "sum up is: " << m_dict[6].rows + m_dict[7].rows + m_dict[17].rows << endl;
}
