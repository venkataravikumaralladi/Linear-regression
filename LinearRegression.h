// ============================================================================
// <copyright file="LinearRegression.h" company="VRK">
//     Copyright (c) Venkata  2020. All rights reserved.
// </copyright>
// ============================================================================

#pragma once

// ============================================================================
// Local includes, e.g. files in the same folder


// ============================================================================
// STL includes.
#include <vector>

// ============================================================================
// Other includes with complete path as Layer/Component/third party...


// ============================================================================
// Forward declarations


// ============================================================================
// Declarations

namespace VRK {

	/** Linear regression with multiple variables: Here we have multiple features 'X'.
	  * if Y is linearly dependent only on X we can use the gradient descent algorithm to find
	  * the equation. Y = thetazero + (thetaone * feataure1) + (thetatwo * feature2) +...+(thetaN * featureN).
	  * Gardient descent algorithm helps us to find parameters theta zero, thetaone, ... and thetaN.
	  * Bried overview of Gradient descent algorithm: (Reference: Prof AndrewNg Machine learning Week1 and Week2.)
	  *
	  * Let us establish notation for variable names for easy understanding.
	  * We will denote the input vector X is of shape (number of examples, 1 + number of features). Here we add dummy
	  * feature x0 as 1 which is used for calcaulating thetazero variable.
	  *
	  *           X = [ [1, eg1Feature1Value, eg1Feature2Value, ..., eg1FeatureNValue],
	  *                 [1, eg2Feature1Value, eg2Feature2Value, ..., eg2FeatureNValue],
	  *                 ...
	  *                 [1, egMFeature1Value, egMFeature2Value, ..., egMFeatureNValue]
	  *               ]
	  *
	  * Ouput value we are interested is Y is shape is (number of examples, 1).
	  *
	  * with training set both X and Y we will find theta variables, and use the equation h(x) to predict for future inputs.
	  * the function h is called a hypothesis.
	  *
	  * Gradient descent algorithm:
	  *
	  *   initialize theata to random numbers. Here in implementation I used zeros.
	  *   presCost = 0.0
	  *   prevCost = 100.0
	  *   while (presCost - prevCost > 0.001)
	  *         prevCost = presCost
	  *         theta = theta - (step/numberofexamples) * direction of movement
	  *         presCost = calcuate cost with new theta values.
	  *
	  * The class implements linear regression algorithm using C++.
	  *
	  */

	class LinearRegression {
	public:
		/** LinearRegression constructor.
		  * @param stepSize used in linear regression algorithm
		  * @return none.
		 */
		LinearRegression(double stepSize = 0.001) {
			m_stepSize = stepSize;
		}

		/** Implements Gradient descent algorithm.
		  * @param X input data independent featrues of shape( number of examples, 1 + number of features)
		  * @param Y inuut data dependent feature i.e, value to be predicted.
		  * @param numberOfIterations. This is used in case convergence did not happen.
		  * @return theta of parameters.
		 */
		std::vector<double> GradientDescentMulipleFeatures(std::vector<std::vector<double>> X,
			std::vector<double> Y,
			int numberOfIterations);


		/** Returns cost vector which can be used for debugging for understading cost. Every
		  * iteration cost should reduce..
		  * @return cost vector.
		 */
		std::vector<double> GetCostVector() { return vecCost; }

	private:
		double m_stepSize;; // determines step size in gradient descent algorithm.
		std::vector<double> vecCost;

		double ComputeCost(std::vector<std::vector<double>> X, std::vector<double> Y, std::vector<double> theta);



	};

}; // namespace 
 
