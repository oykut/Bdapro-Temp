import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD



object TestAdaOptimizer extends App {

  override def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir", "c:/winutil/")
    System.setProperty("spark.sql.warehouse.dir", "file:///C:/spark-warehouse")
    val sc = new SparkContext(new SparkConf().setAppName("TESTADAOPTIMIZER").setMaster("local[*]"))
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    val training = MLUtils.loadLibSVMFile(sc, "data/a9a").repartition(4)
    val testing = MLUtils.loadLibSVMFile(sc, "data/a9at")
   val lr = new LogisticRegressionWithAdaSGD()
    val svm = new SVMWithAdaSGD()

    var updater0 = new SimpleUpdater
    var updater1 = new MomentumUpdater
    var updater2 = new NesterovUpdater
    var updater3 = new AdagradUpdater //0.01 learning rate
    var updater4 = new AdadeltaUpdater // no learning rate
    var updater5 = new RMSpropUpdater // 0.001 learning rate
    var updater6 = new AdamUpdater //0.002
    var updater7 = new AdamaxUpdater //0.002 learning rate
    var updater8 = new NadamUpdater //0.002
    var updater9 = new AMSGradUpdater //0.002 like adam


        lr.optimizer
          .setRegParam(0.0)
          .setNumIterations(1000)
          .setConvergenceTol(0.001)
          .setStepSize(0.1)
          .setUpdater(updater7)
          .setMiniBatchFraction(1)

        val currentTime = System.currentTimeMillis()
        val model = lr.run(training)
        val elapsedTime = System.currentTimeMillis() - currentTime
        // Compute raw scores on the training set.
        val predictionAndLabels = training.map { case LabeledPoint(label, features) =>
          val prediction = model.predict(features)
          (prediction, label)
        }
        // Get evaluation metrics.
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val accuracy = metrics.accuracy
        println(s"Accuracy = $accuracy, time elapsed: $elapsedTime millisecond.")

    training.unpersist()
    sc.stop()
  }
}
