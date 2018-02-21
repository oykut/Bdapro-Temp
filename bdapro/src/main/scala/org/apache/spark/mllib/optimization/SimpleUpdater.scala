package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

class SimpleUpdater extends AdaptiveUpdater {
  def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.asBreeze, brzWeights)

    (Vectors.fromBreeze(brzWeights), 0)
  }
}
