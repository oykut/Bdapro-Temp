package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math._
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, Vector => BV, axpy => brzAxpy, norm => brzNorm , max=> brzMax}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}

class AMSGradUpdater extends AdaptiveUpdater {

  private [this] var squaredGradients: DenseVector[Double] = null
  private [this] var gradients: DenseVector[Double] = null
  private [this] var maxSquaredGradients: DenseVector[Double] = null

  def compute(weightsOld: Vector,
              gradient: Vector,
              stepSize: Double,
              smoothingTerm: Double,
              beta: Double,
              betaS: Double,
              iter: Int,
              regParam : Double): (Vector, Double) = {
    val brzWeights: DenseVector[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: DenseVector[Double] = gradient.asBreeze.toDenseVector
    if(squaredGradients == null) {
      squaredGradients = (1-betaS) * (brzGradient :* brzGradient)
      maxSquaredGradients = squaredGradients
      gradients = (1-beta) * brzGradient
    }
    else {
      squaredGradients = betaS * squaredGradients + (1-betaS) * (brzGradient :* brzGradient)
      maxSquaredGradients = brzMax(squaredGradients,maxSquaredGradients)
      gradients = beta*gradients + (1-beta)* brzGradient
    }
    val denom: DenseVector[Double] = brzSqrt(squaredGradients) + smoothingTerm
    val mult = DenseVector.fill(weightsOld.size){stepSize} / denom
    val update: DenseVector[Double] =  mult :* gradients
    val weightsNew = brzWeights - update
    (Vectors.fromBreeze(weightsNew), 0)
  }

}