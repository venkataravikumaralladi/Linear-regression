int main() {
cout << "Linear regression with Gradient Descent. \n";
		std::streambuf* poldcinbuf = std::cin.rdbuf();

		std::ifstream fileHandle("inDataLR.txt");
		std::cin.set_rdbuf(fileHandle.rdbuf());

		// first line consists of number of features and number of examples
		int numberOfFeatures = 0;
		int numberOfExamples = 0;
		cin >> numberOfFeatures;
		cin >> numberOfExamples;
		cin.ignore(); // ignore the new line.

		// read the features
		vector<vector<double>> vecXData;
		vector<double> vecYData;
		for (unsigned int inputIdx = 0; inputIdx < numberOfExamples; inputIdx++) {
			vector<double> tempFeature;
			tempFeature.push_back(1.0); // think of as basic feature which represents y intercept in line equation.
			for (unsigned int idxFeat = 0; idxFeat < numberOfFeatures; idxFeat++) {
				double currFeatVal = 0.0;
				cin >> currFeatVal;
				tempFeature.push_back(currFeatVal);
			}
			vecXData.push_back(tempFeature);
			double yData;
			cin >> yData;
			vecYData.push_back(yData);
			cin.ignore();
		}

		// Built the regression model.
		VRK::LinearRegression lnReg(0.01);
		vector<double> theta = lnReg.GradientDescentMulipleFeatures(vecXData, vecYData, 100000);

		// now read the prediction inputs
		int iNumberOfPredictions;
		cin >> iNumberOfPredictions;
		cin.ignore();
		vector<double> vecCurrFetVaues;
		for (int idx = 0; idx < iNumberOfPredictions; idx++) {
			vecCurrFetVaues.push_back(1.0);
			for (int fetIdx = 0; fetIdx < numberOfFeatures; fetIdx++) {
				double currFeature;
				cin >> currFeature;
				vecCurrFetVaues.push_back(currFeature);
			}
			cout << std::inner_product(theta.begin(), theta.end(), vecCurrFetVaues.begin(), 0.0) << endl;
			vecCurrFetVaues.clear();
		}
	return 0;
}

/* Example of input file:

2 7
0.18 0.89 109.85
1.0 0.26 155.72
0.92 0.11 137.66
0.07 0.37 76.17
0.85 0.16 139.75
0.99 0.41 162.6
0.87 0.47 151.77
4
0.49 0.18
0.57 0.83
0.56 0.64
0.76 0.18

*/