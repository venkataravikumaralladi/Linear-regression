// ============================================================================
// <copyright file="LinearRegression.cpp" company="VRK">
//     Copyright (c) Venkata  2020. All rights reserved.
// </copyright>
// ============================================================================


// ============================================================================
// Local includes, e.g. files in the same folder, typically corresponding declarations.
#include "LinearRegression.h"

// ============================================================================
// System includes, e.g. STL.
#include <cassert>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <numeric> 
#include <functional>
#include <iomanip>

// ============================================================================
// Other includes with complete path as Layer/Component/third party and so on...


// ============================================================================
// Definitions


// =============================================================================
// namespace declarations;
using namespace std;
using namespace VRK;


double LinearRegression::ComputeCost(std::vector<std::vector<double>> X,
	                                 std::vector<double> Y,
	                                 vector<double> theta) {

	int numberOfExamples = Y.size();
	vector<double> vecPredictions(numberOfExamples, 0);
	/*
		 calculate predictions using below formula
		 ypredction = x0 * theta0 + feature1 * theta1 + .... + featuren * thetan
	*/
	for (unsigned int inputIdx = 0; inputIdx < numberOfExamples; inputIdx++) {
		vecPredictions[inputIdx] = std::inner_product(X[inputIdx].begin(), X[inputIdx].end(), theta.begin(), 0.0);
	}

	double cost = std::inner_product(vecPredictions.begin(), vecPredictions.end(), Y.begin(), 0.0,
		[](auto a, auto b) { return a + b; },
		[](auto a, auto b) { return (a - b) * (a - b); });

	return (cost / (2.0 * numberOfExamples));


}

vector<double> LinearRegression::GradientDescentMulipleFeatures(std::vector<std::vector<double>> X,
																std::vector<double> Y,
	                                                            int numberOfIterations) {
	// check input assumptions.
	assert(X.size() > 0);
	assert(Y.size() > 0);
	assert(X[0].size() > 0);

	int numberOfExamples  = Y.size();
	int numberOfFeatures  = X[0].size();
	int iCurrentIteration = 0;
	double prevCost       = 0.0;
	double currCost       = 100.0;

	vector<double> theta(numberOfFeatures, 0);

	for (; abs(currCost - prevCost)> 0.0001; iCurrentIteration++) {

		prevCost = currCost;
		vector<double> vecPredictions(numberOfExamples, 0);
		/*
			 calculate predictions using below formula
			 ypredction = x0 * theta0 + feature1 * theta1 + .... + featuren * thetan
		*/
		for (unsigned int inputIdx = 0; inputIdx < numberOfExamples; inputIdx++) {
			vecPredictions[inputIdx] = std::inner_product(X[inputIdx].begin(), X[inputIdx].end(), theta.begin(), 0.0);
		}
		
		/* calculate average error for each example.

			avgErrorforfeatureidx = (1/numberofExamples) (summation of all examples of
														(difference between prediction and actual value) * featureidxvalueof that example

		*/
		vector<double> slpDirection(numberOfFeatures, 0);
		for (unsigned int featIdx = 0; featIdx < numberOfFeatures; featIdx++) {
			int idx = 0;
			slpDirection[featIdx] = std::inner_product(vecPredictions.begin(), vecPredictions.end(),
				Y.begin(), 0.0,
				[](auto a, auto b) { return a + b; },
				[&X, &idx, &featIdx](auto a, auto b) { return (a - b) * X[idx++][featIdx]; });

		}
		
		double stepSize = m_stepSize;
		int    idx = 0;
		std::transform(theta.begin(), theta.end(), theta.begin(),
			           [&idx, stepSize, numberOfExamples, &slpDirection](auto a) { return (a - ((stepSize / numberOfExamples) * slpDirection[idx++])); });

		currCost = ComputeCost(X, Y, theta);
		vecCost.push_back(currCost);
	}

	return theta;

}
