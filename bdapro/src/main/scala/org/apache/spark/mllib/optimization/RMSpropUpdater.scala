package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}


class RMSpropUpdater extends AdaptiveUpdater{

  private [this] var accGradient: DenseVector[Double] = null


  def compute(weightsOld: Vector,
              gradient: Vector,
              stepSize: Double,
              smoothingTerm: Double,
              iter: Int,
              regParam : Double): (Vector, Double) = {
    val brzWeights: DenseVector[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: DenseVector[Double] = gradient.asBreeze.toDenseVector

    if (accGradient == null) accGradient = DenseVector.zeros(gradient.size)

    //accumulate gradient
    accGradient = 0.9*accGradient + 0.1 * (brzGradient :* brzGradient)

    //compute update
    val denom: DenseVector[Double] = brzSqrt(accGradient + smoothingTerm)
    val mult =  DenseVector.fill(weightsOld.size){ stepSize }/ denom
    val update: DenseVector[Double] =  mult :* brzGradient

    val weightsNew = brzWeights - update

    (Vectors.fromBreeze(weightsNew), 0)
  }

}