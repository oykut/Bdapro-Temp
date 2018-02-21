package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}


class AdadeltaUpdater extends AdaptiveUpdater{

  private [this] var accGradient: DenseVector[Double] = null
  private [this] var accUpdates: DenseVector[Double] = null
  private [this] var updatesInit: Boolean = false


  def compute(weightsOld: Vector,
              gradient: Vector,
              stepSize: Double,
              smoothingTerm: Double,
              iter: Int,
              rho: Double,
              regParam : Double): (Vector, Double) = {
    val brzWeights: DenseVector[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: DenseVector[Double] = gradient.asBreeze.toDenseVector

    if (accGradient == null) accGradient = DenseVector.zeros(gradient.size)
    if (accUpdates == null) accUpdates = DenseVector.zeros(gradient.size)
    //accumulate gradient
    accGradient = rho*accGradient + (1-rho) * (brzGradient :* brzGradient)
    //compute update
    val update = (brzSqrt(accUpdates + smoothingTerm) / brzSqrt(accGradient + smoothingTerm)) :* brzGradient
    //accumulate updates
    accUpdates = rho*accUpdates + (1-rho) * (update :* update)
    //apply updates
    val weightsNew = brzWeights - update

    (Vectors.fromBreeze(weightsNew), 0)
  }

}
