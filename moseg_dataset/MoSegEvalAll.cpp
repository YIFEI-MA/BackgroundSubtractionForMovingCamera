#include <string.h>
#include <string>
#include <fstream>
#include <CVector.h>
#include <CMatrix.h>

int main(int argc, char* args[]) {
  if (argc <= 3) {
    std::cout << "Usage: MoSegEvalAll shotList.txt 10|50|200|all trackList.txt" << std::endl;
    return -1;
  }
  std::ifstream aShotList(args[1]);
  std::ifstream aTrackList(args[3]);
  if (!aShotList.is_open()) {
    std::cerr << args[1] << " could not be opened." << std::endl;
    return -1;
  }
  if (!aTrackList.is_open()) {
    std::cerr << args[3] << " could not be opened." << std::endl;
    return -1;
  }
  int aEvaluationMode;
  if (strcmp(args[2],"all") == 0) aEvaluationMode = -1;
  else aEvaluationMode = atoi(args[2]);
  if (aEvaluationMode != -1 && aEvaluationMode != 10 && aEvaluationMode != 50 && aEvaluationMode != 200) {
    std::cerr << "Evaluation of " << args[2] << " frames is not allowed. Choose 10, 50, 200, or all frames." << std::endl;
    return -1;
  }
  // Open output file
  std::string s = args[3];
  s.erase(s.find_last_of('.'),s.length());
  s += "Numbers.txt";
  std::ofstream aOut(s.c_str());
  if (!aOut.is_open()) {
    std::cerr << "Could not write output file." << std::endl;
    return -1;
  }
  float aSumDensity = 0.0f;
  float aSumOverallError = 0.0f;
  float aSumAverageError = 0.0f;
  int aSumMergingError = 0;
  int aSumExtracted = 0;
  char dummy[300];
  int aSize;
  aShotList >> aSize; aShotList.getline(dummy,300);
  int aCounter = 0;
  int aCounter2 = 0;
  for (int shot = 0; shot < aSize; shot++) {
    // Read parts of the definition file ---------------------------------------
    std::string aShotLine;
    aShotList >> aShotLine;
    std::ifstream aPropFile(aShotLine.c_str());
    if (!aPropFile.is_open()) {
      std::cerr << "Definition file " << aShotLine << "  not found." << std::endl;
      return -1;
    }
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    // Number of regions
    aPropFile.getline(dummy,300);
    int aRegionNo;
    aPropFile >> aRegionNo; aPropFile.getline(dummy,300);
    // Number of frames
    for (int i = 0; i < 3*aRegionNo; i++)
      aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    aPropFile.getline(dummy,300);
    int aTotalFrameNo;
    aPropFile >> aTotalFrameNo; aPropFile.getline(dummy,300);
    // Ignore this shot if it does not have the required number of frames
    if (aEvaluationMode > 0 && aTotalFrameNo < aEvaluationMode) {
      std::cout << "Only " << aTotalFrameNo << " frames. " << aShotLine << " ignored." << std::endl;
      continue;
    }
    // Remove empty lines in list of tracking files ----------------------------
    std::string aTrackLine;
    do {
      aTrackList >> aTrackLine;
      while (aTrackLine[0] == ' ')
        aTrackLine.erase(0,1);
    } while (aTrackLine.length() == 1);
    // Run evaluation tool -----------------------------------------------------
    std::string s = "./MoSegEval ";
    s += aShotLine + ' ' + aTrackLine;
    std::cout << s.c_str() << std::endl;
    if (system(s.c_str()) != 0) {
      std::cerr << "Error while running " << s << std::endl;
      return -1;
    }
    // Evaluate result file ----------------------------------------------------
    aCounter++;
    s = aTrackLine;
    s.erase(s.find_last_of('.'),s.length());
    std::ifstream aResult((s+"Numbers.txt").c_str());
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    // Number of evaluated frames
    int aEvaluatedFrames;
    aResult.getline(dummy,300);
    aResult >> aEvaluatedFrames; aResult.getline(dummy,300);
    if (aEvaluatedFrames != aEvaluationMode && aEvaluationMode > 0) {
      std::cerr << "The tracks listed in " << args[3] << " have been computed considering different numbers of frames." << std::endl;
      return -1;
    }
    // Density
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    float aDensity;
    aResult >> aDensity; aResult.getline(dummy,300);
    aSumDensity += aDensity;
    if (aDensity > 0) aCounter2++;
    // Pixel error
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    float aOverallError;
    aResult >> aOverallError;
    if (aDensity > 0) aSumOverallError += aOverallError;
    // Average error
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    for (int i = 0; i < 2*aRegionNo; i++)
      aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    float aAverageError;
    aResult >> aAverageError; aResult.getline(dummy,300);
    aSumAverageError += aAverageError;
    // Merging error
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    int aMergingError;
    aResult >> aMergingError; aResult.getline(dummy,300);
    if (aDensity > 0) aSumMergingError += aMergingError;
    // Extracted objects
    aResult.getline(dummy,300);
    aResult.getline(dummy,300);
    int aExtracted;
    aResult >> aExtracted;
    aSumExtracted += aExtracted;
  }
  // Write overall outcome -----------------------------------------------------
  float invSize = 1.0f/aCounter;
  float invSize2 = 1.0f/aCounter2;   // aCounter2 is for results with more than 0 density
  aOut << "Evaluation results for segmentations obtained using ";
  if (aEvaluationMode > 0) aOut << aEvaluationMode;
  else aOut << "all";
  aOut << " frames" << std::endl;
  aOut << "MoSegEval Version 1.0" << std::endl << std::endl;
  aOut << aCounter <<  " shots were evaluated." << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Density (in percent): " << std::endl << aSumDensity*invSize << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Overall (per pixel) clustering error (in percent): " << std::endl << aSumOverallError*invSize2 << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Average (per region) clustering error (in percent): " << std::endl << aSumAverageError*invSize << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Average oversegmentation error: " << std::endl << aSumMergingError*invSize2 << std::endl;
  aOut << "----------------------------" << std::endl;
  aOut << "Objects extracted with less than 10% error (excluding background): " << std::endl << aSumExtracted << std::endl;
  return 0;
}
