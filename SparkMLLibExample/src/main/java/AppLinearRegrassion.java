import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Vector;

public class AppLinearRegrassion {
    public static void main(String[] args) {


        System.setProperty("hadoop.home.dir","//media//alig//Store//hadoop-common-2.2.0-bin-master");

        SparkSession sparkSession = SparkSession.builder().appName("spark-mllib").master("local").getOrCreate();

        Dataset<Row> rawData = sparkSession.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("file//salesb.csv");


        Dataset<Row> newData = sparkSession.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("file//pred.csv");


        VectorAssembler features_vector = new VectorAssembler().setInputCols(new String[]{"Ay"})
                .setOutputCol("features");

        Dataset<Row> transform = features_vector.transform(rawData);

        Dataset<Row> transformNew = features_vector.transform(newData);


        Dataset<Row> finalData = transform.select("features", "Satis");

        Dataset<Row>[] datasets = finalData.randomSplit(new double[]{0.7, 0.3});

        Dataset<Row> trainData=datasets[0];
        Dataset<Row> testData = datasets[1];

        LinearRegression lr = new LinearRegression();

       lr.setLabelCol("Satis");

        LinearRegressionModel model=lr.fit(trainData);

        Dataset<Row> transforTest = model.transform(testData);

        Dataset<Row> transforTestNew = model.transform(transformNew);


        LinearRegressionTrainingSummary summary = model.summary();

        System.out.println(summary.r2());

        transforTest.show();

        transforTestNew.show();



    }
}
