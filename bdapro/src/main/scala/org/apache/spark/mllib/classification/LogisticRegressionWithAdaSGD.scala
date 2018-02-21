/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD


class LogisticRegressionWithAdaSGD private[mllib](
                                                 private var stepSize: Double,
                                                 private var numIterations: Int,
                                                 private var regParam: Double,
                                                 private var miniBatchFraction: Double,
                                                 private var updater: AdaptiveUpdater)
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  private val gradient = new LogisticGradient()

  override val optimizer = new GradientDescentAlg(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)

  override protected val validators = List(DataValidators.binaryLabelValidator)
  /**
    * Construct a LogisticRegression object with default parameters: {stepSize: 1.0,
    * numIterations: 100, regParm: 0.01, miniBatchFraction: 1.0}.
    */
  @Since("2.1.0")
  def this() = this(1.0, 100, 0.01, 1.0, new AdagradUpdater)

  override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}

/**
  * Top-level methods for calling Logistic Regression.
  * NOTE: Labels used in Logistic Regression should be {0, 1}
  */
@Since("2.1.0")
@deprecated("Use ml.classification.LogisticRegression or LogisticRegressionWithLBFGS", "2.0.0")
object LogisticRegressionWithAdaSGD {
  // NOTE(shivaram): We use multiple train methods instead of default arguments to support
  // Java programs.

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. Each iteration uses
    * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
    * gradient descent are initialized using the initial weights provided.
    * NOTE: Labels used in Logistic Regression should be {0, 1}
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.
    * @param miniBatchFraction Fraction of data to be used per iteration.
    * @param initialWeights Initial set of weights to be used. Array should be equal in size to
    *        the number of features in the data.
    */
  @Since("2.1.0")
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double,
             initialWeights: Vector,
             updater: AdaptiveUpdater): LogisticRegressionModel = {
    new LogisticRegressionWithAdaSGD(stepSize, numIterations, 0.0, miniBatchFraction, updater)
      .run(input, initialWeights)
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. Each iteration uses
    * `miniBatchFraction` fraction of the data to calculate the gradient.
    * NOTE: Labels used in Logistic Regression should be {0, 1}
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @param stepSize Step size to be used for each iteration of gradient descent.

    * @param miniBatchFraction Fraction of data to be used per iteration.
    */
  @Since("2.1.0")
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double,
             updater: AdaptiveUpdater): LogisticRegressionModel = {
    new LogisticRegressionWithAdaSGD(stepSize, numIterations, 0.0, miniBatchFraction, updater)
      .run(input)
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using the specified step size. We use the entire data
    * set to update the gradient in each iteration.
    * NOTE: Labels used in Logistic Regression should be {0, 1}
    *
    * @param input RDD of (label, array of features) pairs.
    * @param stepSize Step size to be used for each iteration of Gradient Descent.

    * @param numIterations Number of iterations of gradient descent to run.
    * @return a LogisticRegressionModel which has the weights and offset from training.
    */
  @Since("2.1.0")
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             updater: AdaptiveUpdater): LogisticRegressionModel = {
    train(input, numIterations, stepSize, 1.0, updater)
  }

  /**
    * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
    * number of iterations of gradient descent using a step size of 1.0. We use the entire data set
    * to update the gradient in each iteration.
    * NOTE: Labels used in Logistic Regression should be {0, 1}
    *
    * @param input RDD of (label, array of features) pairs.
    * @param numIterations Number of iterations of gradient descent to run.
    * @return a LogisticRegressionModel which has the weights and offset from training.
    */
  @Since("2.1.0")
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             updater: AdaptiveUpdater): LogisticRegressionModel = {
    train(input, numIterations, 1.0, 1.0, updater)
  }
}

