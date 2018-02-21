import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD



object TestLoopAdaOptimizer extends App {

  override def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir", "c:/winutil/")
    System.setProperty("spark.sql.warehouse.dir", "file:///C:/spark-warehouse")
    val sc = new SparkContext(new SparkConf().setAppName("TESTADAOPTIMIZER").setMaster("local[*]"))
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    val training = MLUtils.loadLibSVMFile(sc, "data/a9a").repartition(4)
    val testing = MLUtils.loadLibSVMFile(sc, "data/a9at")

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


    val updaters = Seq(updater0, updater1, updater2, updater3, updater4, updater5, updater6,
      updater7, updater8, updater9)

    val rates = Seq(1, 1, 0.1, 0.01, 1, 0.001, 0.02, 0.02, 0.02, 0.02)

    for (i <- 0 to 0){
      if (i==0){ val lr = new LogisticRegressionWithAdaSGD()
        var u = 0
        for(updater<-updaters){
         // for (r <- Seq(900,400,90,40,9,4,0,-0.5,-0.9,-0.99,-0.999)) {
          //for (r <- Seq(90,40,9,4,0,-0.5,-0.9)) {
          for (r <- Seq(9,0,-0.9)) {
            val rate = rates(u) + rates(u) * r
            lr.optimizer
              .setRegParam(0.0)
              .setNumIterations(500)
              .setConvergenceTol(0.001)
              .setStepSize(rate)
              .setUpdater(updater).
              setMiniBatchFraction(0.5)
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
            println(s"Accuracy  of updater $u on alg $i with rate $rate = $accuracy, time elapsed: $elapsedTime millisecond.")
          }
          u=u+1
        }
      }
      else {
        val svm = new SVMWithAdaSGD()
        var u = 0
        for (updater <- updaters) {
          for (r <- Seq(900,400,90,40,9,4,0,-0.5,-0.9,-0.99,-0.999)) {
            val rate = rates(u) + rates(u) * r
            svm.optimizer
              .setRegParam(0.0)
              .setNumIterations(400)
              .setConvergenceTol(0.001)
              .setStepSize(rate)
              .setUpdater(updater).
              setMiniBatchFraction(1)
            val currentTime = System.currentTimeMillis()
            val model = svm.run(training)
            val elapsedTime = System.currentTimeMillis() - currentTime
            // Compute raw scores on the training set.
            val predictionAndLabels = training.map { case LabeledPoint(label, features) =>
              val prediction = model.predict(features)
              (prediction, label)
            }
            // Get evaluation metrics.
            val metrics = new MulticlassMetrics(predictionAndLabels)
            val accuracy = metrics.accuracy
            println(s"Accuracy  of updater $u on alg $i with rate $rate = $accuracy, time elapsed: $elapsedTime millisecond.")
          }
          u = u + 1
        }
      }
    }
    training.unpersist()
    sc.stop()
  }
}
