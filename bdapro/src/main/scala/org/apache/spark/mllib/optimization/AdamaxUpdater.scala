package org.apache.spark.mllib.optimization


import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math._
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, Vector => BV, axpy => brzAxpy, norm => brzNorm, max}
import breeze.numerics.{sqrt => brzSqrt, abs}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}

class AdamaxUpdater extends AdaptiveUpdater {


  private [this] var squaredGradients: DenseVector[Double] = null
  private [this] var gradients: DenseVector[Double] = null
  private [this] var betaPower: Double = 0
  private [this] var u: DenseVector[Double] = null
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
      gradients = (1-beta) * brzGradient
      betaPower = beta
      u=abs(brzGradient)
    }
    else {
      squaredGradients = betaS * squaredGradients + (1-betaS) * (brzGradient :* brzGradient)
      gradients = beta*gradients + (1-beta)* brzGradient
      betaPower = betaPower * beta
      u = max(betaS*u,abs(brzGradient))
    }
    val m =  (1/(1-betaPower)) * gradients
    val mult = DenseVector.fill(weightsOld.size){stepSize} / u
    val update: DenseVector[Double] =  mult :* m
    val weightsNew = brzWeights - update
    (Vectors.fromBreeze(weightsNew), 0)
  }

}
