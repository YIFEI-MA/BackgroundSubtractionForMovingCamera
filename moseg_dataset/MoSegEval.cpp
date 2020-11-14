#include <algorithm>
#include <string.h>
#include <CVector.h>
#include <CMatrix.h>

class CPoint {
public:
  CPoint() {}
  int x,y,frame;
};

class CSimpleTrack {
public:
  CSimpleTrack() {}
  int mLabel;
  CVector<CPoint> mPoints;
};

class CCoverage {
public:
  CCoverage() {}
  CCoverage(int aRegion, int aCoverage) : mRegion(aRegion),mCoverage(aCoverage) {}
  int mRegion;
  int mCoverage;
};

bool operator<(const CCoverage& a, const CCoverage& b) {
  return a.mCoverage < b.mCoverage;
}

int mTotalFrameNo;
int mLabeledFramesNo;
int mRegionNo;
int mClusterNo;

CVector<int> mLabeledFrames;
CVector<CMatrix<float> > mRegions;
int mEvaluateFrames;
CVector<CSimpleTrack> mTracks;
CVector<int> mColor2Region;

bool readTracks(char* aFilename) {
  std::ifstream aFile(aFilename);
  if (!aFile.is_open()) {
    std::cerr << aFilename << " was not found." << std::endl;
    return false;
  }
  int aLength;
  aFile >> aLength;
  if (aLength == 10) mEvaluateFrames = 10;
  else if (aLength == 50) mEvaluateFrames = 50;
  else if (aLength == 200) mEvaluateFrames = 200;
  else if (aLength == mTotalFrameNo) mEvaluateFrames == mTotalFrameNo;
  else {
    std::cerr << "Length of tracked sequence does not match a typical evaluation length (10,50,200,all frames)." << std::endl;
    return false;
  }
  int aTrackNo;
  aFile >> aTrackNo;
  mTracks.setSize(aTrackNo);
  for (int i = 0; i < aTrackNo; i++) {
    aFile >> mTracks(i).mLabel;
    int aSize;
    aFile >> aSize;
    mTracks(i).mPoints.setSize(aSize);
    float x,y,frame;
    for (int j = 0; j < aSize; j++) {
      aFile >> x >> y >> frame;
      mTracks(i).mPoints(j).x = (int)(x+0.5f);
      mTracks(i).mPoints(j).y = (int)(y+0.5f);
      mTracks(i).mPoints(j).frame = (int)frame;
    }
  }
  // Count number of clusters
  mClusterNo = 0;
  for (int i = 0; i < mTracks.size(); i++)
    if (mTracks(i).mLabel+1 > mClusterNo) mClusterNo = mTracks(i).mLabel+1;
  return true;
}

int main(int argc, char* args[]) {
  if (argc <= 2) {
    std::cout << "Usage: MoSegEval shotDef.dat yourtracks.dat" << std::endl;
    return -1;
  }
  // Read definition file and ground truth -------------------------------------
  std::string s = args[1];
  std::string aPath = s;
  if (aPath.find_last_of('/') < aPath.length()) aPath.erase(aPath.find_last_of('/'),aPath.length());
  else aPath = ".";
  aPath += '/';
  std::ifstream aPropFile(s.c_str());
  if (!aPropFile.is_open()) {
    std::cerr << "Definition file " << s.c_str() << "  not found." << std::endl;
    return -1;
  }
  // Read header
  char dummy[300];
  aPropFile.getline(dummy,300);
  aPropFile.getline(dummy,300);
  // Number of regions
  aPropFile.getline(dummy,300);
  aPropFile >> mRegionNo; aPropFile.getline(dummy,300);
  mColor2Region.setSize(256);
  CMatrix<float> aPenalty(mRegionNo,mRegionNo);
  // Region color
  for (int i = 0; i < mRegionNo; i++) {
    aPropFile.getline(dummy,300);
    int a;
    aPropFile >> a; aPropFile.getline(dummy,300);
    mColor2Region(a) = i;
  }
  // Confusion penalty matrix
  aPropFile.getline(dummy,300);
  aPropFile.getline(dummy,300);
  for (int j = 0; j < mRegionNo; j++)
    for (int i = 0; i < mRegionNo; i++)
      aPropFile >> aPenalty(i,j);
  // Number of frames in shot
  aPropFile.getline(dummy,300);
  aPropFile.getline(dummy,300);
  aPropFile.getline(dummy,300);
  aPropFile >> mTotalFrameNo; aPropFile.getline(dummy,300);
  // Number of labeled frames
  aPropFile.getline(dummy,300);
  aPropFile >> mLabeledFramesNo; aPropFile.getline(dummy,300);
  mLabeledFrames.setSize(mLabeledFramesNo);
  mRegions.setSize(mLabeledFramesNo);
  // Read frame number and annotation
  for (int i = 0; i < mLabeledFramesNo; i++) {
    aPropFile.getline(dummy,300);
    aPropFile >> mLabeledFrames(i); aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    std::string s;
    aPropFile >> s;
    mRegions(i).readFromPGM((aPath+s).c_str());
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
  }
  // Read tracks ---------------------------------------------------------------
  if (!readTracks(args[2])) return -1;
  // Evaluate ------------------------------------------------------------------
  CMatrix<int> aRegionClusterOverlap(mRegionNo,mClusterNo,0);
  CVector<std::vector<int> > aAssignCluster(mClusterNo);
  CVector<int> aClusterSize(mClusterNo,0);
  CVector<int> aRegionSize(mRegionNo,0);
  int aUsedLabeledFrames = 0;
  int aDoubleOccupation = 0;
  // Measure coverage of regions (ground truth) by clusters (estimated track labels)
  for (int t = 0; t < mLabeledFramesNo; t++) {
    if (mLabeledFrames(t) > mEvaluateFrames && mEvaluateFrames > 0) break;
    aUsedLabeledFrames++;
    CMatrix<bool> aOccupied(mRegions(t).xSize(),mRegions(t).ySize(),false);
    for (int i = 0; i < mRegions(t).size(); i++)
      aRegionSize(mColor2Region((int)(mRegions(t).data()[i])))++;
    for (int i = 0; i < mTracks.size(); i++) {
      if (mTracks(i).mPoints(0).frame > mLabeledFrames(t)
        || mTracks(i).mPoints(mTracks(i).mPoints.size()-1).frame < mLabeledFrames(t)) continue;
      int t2 = mLabeledFrames(t)-mTracks(i).mPoints(0).frame;
      int x = mTracks(i).mPoints(t2).x;
      int y = mTracks(i).mPoints(t2).y;
      if (x < 0 || y < 0 || x >= mRegions(t).xSize() || y >= mRegions(t).ySize()) continue;
      int aRegion = mColor2Region((int)mRegions(t)(x,y));
      aRegionClusterOverlap(aRegion,mTracks(i).mLabel)++;
      aClusterSize(mTracks(i).mLabel)++;
      // Count double occupation of pixels, so it does not increase the density
      if (aOccupied(x,y)) aDoubleOccupation++;
      aOccupied(x,y) = true;
    }
  }
  // Order regions by their total coverage
  std::vector<CCoverage> aOrderedRegions;
  for (int i = 0; i < mRegionNo; i++) {
    int aCoverage = 0;
    for (int j = 0; j < mClusterNo; j++)
      aCoverage += aRegionClusterOverlap(i,j);
    aOrderedRegions.push_back(CCoverage(i,aCoverage));
  }
  std::sort(aOrderedRegions.begin(),aOrderedRegions.end());
  // Go through regions, the one with most coverage first
  for (int i = mRegionNo-1; i >= 0; i--) {
    // Assign the cluster that best covers this region
    int bestj = -1;
    int bestCoverage = 0;
    for (int j = 0; j < mClusterNo; j++) {
      int val = aRegionClusterOverlap(i,j);
      if (aRegionClusterOverlap(i,j) > bestCoverage) {
        bestCoverage = aRegionClusterOverlap(i,j);
        bestj = j;
      }
    }
    if (bestj >= 0) aAssignCluster(bestj).push_back(i);
  }
  // Now counting positives and negatives for each region
  CVector<int> aPositives(mRegionNo,0);
  CVector<float> aWeightedNegatives(mRegionNo,0);
  CVector<int> aNegatives(mRegionNo,0);
  int aOversegmentationPenalty = 0;
  for (int j = 0; j < mClusterNo; j++) {
    if (aClusterSize(j) == 0) continue;
    // 1. cluster assigned exactly once (correct)
    if (aAssignCluster(j).size() == 1) {
      int i = aAssignCluster(j)[0];
      aPositives(i) += aRegionClusterOverlap(i,j);
      // Some points of this cluster may cover other regions -> negatives
      for (int i2 = 0; i2 < mRegionNo; i2++) {
        aWeightedNegatives(i2) += aRegionClusterOverlap(i2,j)*aPenalty(i,i2);
        if (i2 != i) aNegatives(i2) += aRegionClusterOverlap(i2,j);
      }
    }
    // 2. cluster assigned multiple times (undersegmentation)
    else if (aAssignCluster(j).size() > 1) {
      // Find the most covered region
      int best = 0;
      int i = 0;
      for (unsigned int k = 0; k < aAssignCluster(j).size(); k++)
        if (aRegionClusterOverlap(aAssignCluster(j)[k],j) > best) {
          best = aRegionClusterOverlap(aAssignCluster(j)[k],j);
          i = aAssignCluster(j)[k];
        }
      // This region counts positive
      aPositives(i) += aRegionClusterOverlap(i,j);
      // All points assigned to other regions count negative
      for (int i2 = 0; i2 < mRegionNo; i2++) {
        aWeightedNegatives(i2) += aRegionClusterOverlap(i2,j)*aPenalty(i,i2);
        if (i2 != i) aNegatives(i2) += aRegionClusterOverlap(i2,j);
      }
    }
    // 3. cluster not assigned at all (oversegmentation)
    else {
      // Increase the counter for oversegmented regions
      aOversegmentationPenalty++;
      // Find the most covered region
      int best = 0;
      int i = 0;
      for (int i2 = 0; i2 < mRegionNo; i2++)
        if (aRegionClusterOverlap(i2,j) > best) {
          best = aRegionClusterOverlap(i2,j);
          i = i2;
        }
      // This region counts positive
      aPositives(i) += aRegionClusterOverlap(i,j);
      // All points assigned to other regions count negative
      for (int i2 = 0; i2 < mRegionNo; i2++) {
        aWeightedNegatives(i2) += aRegionClusterOverlap(i2,j)*aPenalty(i,i2);
        if (i2 != i) aNegatives(i2) += aRegionClusterOverlap(i2,j);
      }
    }
  }
  // Compute final numbers
  int aTotalCoverage = 0;
  for (int i = 0; i < aRegionClusterOverlap.size(); i++)
    aTotalCoverage += aRegionClusterOverlap.data()[i];
  aTotalCoverage -= aDoubleOccupation;
  float aDensity = 100.0f*aTotalCoverage/(aUsedLabeledFrames*mRegions(0).size());
  int aTotalPositives = 0;
  int aTotalNegatives = 0;
  float aTotalWeightedNegatives = 0;
  for (int i = 0; i < mRegionNo; i++) {
    aTotalPositives += aPositives(i);
    aTotalNegatives += aNegatives(i);
    aTotalWeightedNegatives += aWeightedNegatives(i);
  }
  float aOverallError = 0.0f;
  if (aTotalNegatives+aTotalPositives > 0) aOverallError = 100.0f*aTotalWeightedNegatives/(aTotalNegatives+aTotalPositives);
  CVector<float> aPerRegionError(mRegionNo);
  float aAverageError = 0;
  int aVisibleRegions = mRegionNo;
  int aCoveredRegions = 0;
  int aLessThan10Percent = 0;
  for (int i = 0; i < mRegionNo; i++) {
    // Do not count regions not visible in the evaluated part of the sequence
    if (aRegionSize(i) == 0) {
      aVisibleRegions--;
      continue;
    }
    // Regions covered by at last one cluster
    if (aPositives(i) > 0) {
      aPerRegionError(i) = 100.0f*aNegatives(i)/(aNegatives(i)+aPositives(i));
      aCoveredRegions++;
    }
    // Regions not covered at all: 100% error
    else aPerRegionError(i) = 100.0f;
    aAverageError += aPerRegionError(i);
    if (aPerRegionError(i) < 10) aLessThan10Percent++;
  }
  aAverageError *= 1.0f/aVisibleRegions;
  if (aLessThan10Percent > 0) aLessThan10Percent -= 1;
  // Output results ------------------------------------------------------------
  s = args[2];
  s.erase(s.find_last_of('.'),s.length());
  std::ofstream aOut((s+"Numbers.txt").c_str());
  aOut << "Evaluation results for: " << args[2] << std::endl;
  aOut << "MoSegEval Version 1.0" << std::endl << std::endl;
  aOut << "Number of frames used from the sequence:" << std::endl;
  aOut << mEvaluateFrames << std::endl;
  aOut << "Number of labeled frames in this time window:" << std::endl;
  aOut << aUsedLabeledFrames << std::endl;
  aOut << "--------------------------" << std::endl;
  aOut << "Density (in percent):" << std::endl;
  aOut << aDensity << std::endl;
  aOut << "--------------------------" << std::endl;
  aOut << "Overall (per pixel) clustering error (in percent):" << std::endl;
  aOut << aOverallError << std::endl;
  aOut << "--------------------------" << std::endl;
  aOut << "Clustering error per region (in percent):" << std::endl;
  for (int i = 0; i < mRegionNo; i++) {
    aOut << "Region " << i << ": " << std::endl;
    if (aRegionSize(i) == 0) aOut << "not visible" << std::endl;
    else aOut << aPerRegionError(i) << std::endl;
  }
  aOut << "Visible regions in the evaluated part of the shot:" << std::endl;
  aOut << aVisibleRegions << std::endl;
  aOut << "--------------------------" << std::endl;
  aOut << "Average (per region) clustering error (in percent):" << std::endl;
  aOut << aAverageError << std::endl;
  aOut << "--------------------------" << std::endl;
  aOut << "Number of clusters merged to obtain this result (oversegmentation error):" << std::endl;
  aOut << aOversegmentationPenalty << std::endl;
  aOut << "--------------------------" << std::endl;
  aOut << "Number of regions with less than 10% error (excluding background):" << std::endl;
  aOut << aLessThan10Percent << std::endl;
  return 0;
}
