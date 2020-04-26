import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class AppNaiveBayes {
    public static void main(String[] args) {

        SparkSession sparkSession= new SparkSession.Builder().master("local").appName("spark ml").getOrCreate();


        Dataset<Row> rawData = sparkSession.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("file//basketball.csv");

        // you can do it using loop
        StringIndexer indexHava=new StringIndexer().setInputCol("Hava").setOutputCol("hava_cat");
        StringIndexer indexNem=new StringIndexer().setInputCol("Nem").setOutputCol("nem_cat");
        StringIndexer indexSicaklik=new StringIndexer().setInputCol("Sicaklik").setOutputCol("sicaklik_cat");
        StringIndexer indexRuzgar=new StringIndexer().setInputCol("Ruzgar").setOutputCol("ruzgar_cat");
        StringIndexer indexLabel=new StringIndexer().setInputCol("BOynama").setOutputCol("label");

        Dataset<Row> transformHava = indexHava.fit(rawData).transform(rawData);
        Dataset<Row> transformNem = indexNem.fit(transformHava).transform(transformHava);
        Dataset<Row> transformRuz = indexRuzgar.fit(transformNem).transform(transformNem);
        Dataset<Row> transformSic = indexSicaklik.fit(transformRuz).transform(transformRuz);
        Dataset<Row> transformResult = indexLabel.fit(transformSic).transform(transformSic);

        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(new String[]{"hava_cat","nem_cat","sicaklik_cat","ruzgar_cat","label"} ).setOutputCol("features");

        Dataset<Row> transform = vectorAssembler.transform(transformResult);

        Dataset<Row> finalData = transform.select("label", "features");
        Dataset<Row>[] datasets = finalData.randomSplit(new double[]{0.7, 0.3});

        Dataset<Row> trainData = datasets[0];

        Dataset<Row> testnData = datasets[1];

        NaiveBayes nb = new NaiveBayes();
        nb.setSmoothing(1);
        NaiveBayesModel nModel=nb.fit(trainData);
        Dataset<Row> prediction = nModel.transform(testnData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        Double evaluate=evaluator.evaluate(prediction);

        System.out.println("Accuracy = " + evaluate);

        prediction.show();


        //rawData.show();




    }
}
