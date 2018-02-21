package org.apache.spark.mllib.classification

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

class SVMWithAdaSGD(private var stepSize: Double,
                     private var numIterations: Int,
                     private var regParam: Double,
                     private var miniBatchFraction: Double,
                     private var updater: AdaptiveUpdater) extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {


  private val gradient = new HingeGradient()

  override val optimizer = new GradientDescentAlg(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  /**
    * Construct a SVM object with default parameters: {stepSize: 1.0, numIterations: 100,
    * regParm: 0.01, miniBatchFraction: 1.0}.
    */
  def this() = this(1.0, 100, 0.01, 1.0, new AdagradUpdater)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new SVMModel(weights, intercept)
  }
}

  /**
    * Top-level methods for calling SVM.
    *
    * @note Labels used in SVM should be {0, 1}.
    */
  @Since("0.8.0")
  object SVMWithAdaSGD {

    /**
      * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
      * of iterations of gradient descent using the specified step size. Each iteration uses
      * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
      * gradient descent are initialized using the initial weights provided.
      *
      * @param input RDD of (label, array of features) pairs.
      * @param numIterations Number of iterations of gradient descent to run.
      * @param stepSize Step size to be used for each iteration of gradient descent.
      * @param regParam Regularization parameter.
      * @param miniBatchFraction Fraction of data to be used per iteration.
      * @param initialWeights Initial set of weights to be used. Array should be equal in size to
      *        the number of features in the data.
      *
      * @note Labels used in SVM should be {0, 1}.
      */
    @Since("0.8.0")
    def train(
               input: RDD[LabeledPoint],
               numIterations: Int,
               stepSize: Double,
               regParam: Double,
               miniBatchFraction: Double,
               initialWeights: Vector,
               updater: AdaptiveUpdater): SVMModel = {
      new SVMWithAdaSGD(stepSize, numIterations, regParam, miniBatchFraction, updater)
        .run(input, initialWeights)
    }

    /**
      * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
      * of iterations of gradient descent using the specified step size. Each iteration uses
      * `miniBatchFraction` fraction of the data to calculate the gradient.
      *
      * @note Labels used in SVM should be {0, 1}
      *
      * @param input RDD of (label, array of features) pairs.
      * @param numIterations Number of iterations of gradient descent to run.
      * @param stepSize Step size to be used for each iteration of gradient descent.
      * @param regParam Regularization parameter.
      * @param miniBatchFraction Fraction of data to be used per iteration.
      */
    @Since("0.8.0")
    def train(
               input: RDD[LabeledPoint],
               numIterations: Int,
               stepSize: Double,
               regParam: Double,
               miniBatchFraction: Double,
               updater: AdaptiveUpdater): SVMModel = {
      new SVMWithAdaSGD(stepSize, numIterations, regParam, miniBatchFraction, updater).run(input)
    }

    /**
      * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
      * of iterations of gradient descent using the specified step size. We use the entire data set to
      * update the gradient in each iteration.
      *
      * @param input RDD of (label, array of features) pairs.
      * @param stepSize Step size to be used for each iteration of Gradient Descent.
      * @param regParam Regularization parameter.
      * @param numIterations Number of iterations of gradient descent to run.
      * @return a SVMModel which has the weights and offset from training.
      *
      * @note Labels used in SVM should be {0, 1}
      */
    @Since("0.8.0")
    def train(
               input: RDD[LabeledPoint],
               numIterations: Int,
               stepSize: Double,
               regParam: Double,
               updater: AdaptiveUpdater): SVMModel = {
      this.train(input, numIterations, stepSize, regParam, 1.0, updater)
    }

    /**
      * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
      * of iterations of gradient descent using a step size of 1.0. We use the entire data set to
      * update the gradient in each iteration.
      *
      * @param input RDD of (label, array of features) pairs.
      * @param numIterations Number of iterations of gradient descent to run.
      * @return a SVMModel which has the weights and offset from training.
      *
      * @note Labels used in SVM should be {0, 1}
      */
    @Since("0.8.0")
    def train(input: RDD[LabeledPoint], numIterations: Int,updater: AdaptiveUpdater): SVMModel = {
      train(input, numIterations, 1.0, 0.01, 1.0, updater)
    }
}
