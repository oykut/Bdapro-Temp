package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.math._
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}

class AdagradUpdater extends AdaptiveUpdater {

  private [this] var squaredGradients: DenseVector[Double] = null

  def compute(weightsOld: Vector,
              gradient: Vector,
              stepSize: Double,
              smoothingTerm: Double,
              iter: Int,
              regParam : Double): (Vector, Double) = {
    val brzWeights: DenseVector[Double] = weightsOld.asBreeze.toDenseVector
    val brzGradient: DenseVector[Double] = gradient.asBreeze.toDenseVector
    if(squaredGradients == null) squaredGradients = brzGradient :* brzGradient
    else squaredGradients = squaredGradients + (brzGradient :* brzGradient)
    val denom: DenseVector[Double] = squaredGradients + smoothingTerm
    val root=brzSqrt(denom)
    val mult = DenseVector.fill(weightsOld.size){stepSize} / root
    /*Even though the equation shows a multiplication of a diagonal matrix and a svector,
  here it is transformed to an element-wise multiplication of the diagonal (in vector representation)
  and the vector. The operation is equivalent and it saves memory space and computation time*/
    val update: DenseVector[Double] =  mult :* brzGradient
    val weightsNew = brzWeights - update

    (Vectors.fromBreeze(weightsNew), 0)
  }

}

