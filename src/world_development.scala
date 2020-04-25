import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.when
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.functions._

object world_development {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("World Development")
      .master("local[*]")
      .getOrCreate

    import spark.implicits._

    val data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("file:\\C:\\Programming\\ml_data\\world_development\\WDIData.csv")

    // predict GDP per capita (NY.GDP.PCAP.PP.KD) (this is PPP and inflation adjusted I think)-> Find some good inputs - Build a neural network
    val gdp_per_capita = data.filter($"Indicator Code" === "NY.GDP.PCAP.PP.KD")

    // make code that loops and grabs a given years data and then stitches it together into larger dataset here
    //val years = Array("2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018")
    val years = Array("2015")   // if I just want to do one :)

    val colSeq = Seq("label","Total Employment %","Urban Population %", "Population 0-14 %")
    var full_df = Seq.empty[(Int, Double, Double, Double)].toDF(colSeq:_*)

    for (y <- years) {
      var gdp_given_year = gdp_per_capita.select("Country Name", "Country Code", y).withColumnRenamed(y,"Income")

      // make grouping column: -- Low income = <10000 = 0, Mid income = 10000-45000 = 1, High Income = >45000 = 2
      gdp_given_year = gdp_given_year.withColumn("Income Grouping", when($"Income" < 10000, value = 0).when($"Income" < 45000, value = 1).otherwise(value = 2))

      /*
      // Ideally want a similar amount of case 0,1,2 so good data size for each (my values seem okay)
      gdp_given_year.groupBy("Income Grouping").count().show()
      */

      // selecting factor datasets
      val pop_0_to_14 = data.filter($"Indicator Code" === "SP.POP.0014.TO.ZS").select("Country Code", y).filter(col(y).isNotNull).withColumnRenamed(y,"Population 0-14 %")
      val total_employment = data.filter($"Indicator Code" === "SL.EMP.TOTL.SP.NE.ZS").select("Country Code", y).filter(col(y).isNotNull).withColumnRenamed(y,"Total Employment %")
      val urban_population = data.filter($"Indicator Code" === "SP.URB.TOTL.IN.ZS").select("Country Code", y).filter(col(y).isNotNull).withColumnRenamed(y,"Urban Population %")

      // join together our info and factors for the classifier then select columns care about
      val full_data = gdp_given_year.join(total_employment, "Country Code").join(urban_population, "Country Code").join(pop_0_to_14, "Country Code")

      // grab only the label and indexes we care about
      val df = full_data.select(full_data("Income Grouping").as("label"), $"Total Employment %", $"Urban Population %", $"Population 0-14 %")
      full_df = full_df.union(df)
    }

    // check larger dataset worked properly
    full_df.count()

    // turn the stats we care about into features to be fed into classification model
    import org.apache.spark.ml.feature.{VectorAssembler}
    val assembler = new VectorAssembler().setInputCols(Array("Total Employment %", "Urban Population %", "Population 0-14 %")).setOutputCol("features")

    // transformed df into one with label and features (exciting!)
    val assembled = assembler.transform(full_df)
    assembled.show()

    // split data into training and test set
    val Array(training, test) = assembled.select("label","features").
      randomSplit(Array(0.7, 0.3), seed = 12345)

    // classify it! -- Use random forest first since "simpler"
    import org.apache.spark.ml.classification.RandomForestClassifier
    import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}
    val rf = new RandomForestClassifier()

    // create the param grid
    val paramGrid = new ParamGridBuilder().
      addGrid(rf.numTrees,Array(20,50,100)).
      build()

    // create cross val object, define scoring metric
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    val cv = new CrossValidator().
      setEstimator(rf).
      setEvaluator(new MulticlassClassificationEvaluator().setMetricName("weightedRecall")).
      setEstimatorParamMaps(paramGrid).
      setNumFolds(3).
      setParallelism(2)

    // You can then treat this object as the model and use fit on it.
    val model = cv.fit(training)

    // test it out!
    val results = model.transform(test).select("features", "label", "prediction")
    val predictionAndLabels = results.
      select($"prediction",$"label").
      as[(Double, Double)].
      rdd

    // Instantiate a new metrics objects
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
    val bMetrics = new BinaryClassificationMetrics(predictionAndLabels)
    val mMetrics = new MulticlassMetrics(predictionAndLabels)
    val labels = mMetrics.labels

    // Print out the Confusion matrix
    println("Confusion matrix:    (predicted V vs Actual >)")
    println(mMetrics.confusionMatrix)

    // testing again but now using a multilayer perceptron Neural network
    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    // 3 inputs (features) then 2 mid layers to classify then 3 outputs for classes (0,1,2)
    val layers = Array[Int](3, 20, 20, 3)

    // create the trainer and set its parameters
    val trainer = (new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100))

    // train the model
    val modelNN = trainer.fit(training)

    // test it and compute accuracy
    val resultsNN = modelNN.transform(test)
    val predictionAndLabelsNN = resultsNN.select("prediction", "label")

    // borrowing output stuff from above
    // predictionAndLabelsNN is a Row -> convert this to an RDD of tuples for better analysis
    val NN_RDD = predictionAndLabelsNN.map(row => (row.getDouble(0), row.getInt(1).toDouble)).rdd
    val mMetricsNN = new MulticlassMetrics(NN_RDD)
    // Print out the Confusion matrix
    println("Confusion matrix:    (predicted V vs Actual >)")
    println(mMetricsNN.confusionMatrix)

    // basic accuracy eval stuff
    val evaluator = (new MulticlassClassificationEvaluator()
      .setMetricName("accuracy"))

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabelsNN)}")
  }
}
