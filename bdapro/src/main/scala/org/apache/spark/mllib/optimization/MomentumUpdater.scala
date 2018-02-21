package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.math._
import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}

class MomentumUpdater extends AdaptiveUpdater{

  private [this] var momentumOld: BV[Double] = null

  def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        momentumFraction: Double,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    if(momentumOld == null) {momentumOld = DenseVector.zeros[Double](weightsOld.size)}
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: BV[Double] = gradient.asBreeze.toDenseVector
    val momentumNew = momentumOld * momentumFraction + brzGradient * stepSize
    val weightsNew = brzWeights - momentumNew
    momentumOld = momentumNew
    (Vectors.fromBreeze(weightsNew), 0)
  }
}
