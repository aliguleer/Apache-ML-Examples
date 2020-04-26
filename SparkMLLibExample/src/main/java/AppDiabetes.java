import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AppDiabetes {

    public static void main(String[] args) {


        SparkSession sparkSession = SparkSession.builder().appName("spark-mllib").master("local").getOrCreate();

        Dataset<Row> rawData = sparkSession.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("file//diabetes.csv");


        String[] headerList={"preg","plas","pres","skin","insu","mass","pedi","age","class"};
        List<String> headers= Arrays.asList(headerList);
        List<String> headersRusult= new ArrayList<String>();

        for (String h:headers){
            if(h.equals("class")){
                StringIndexer indexTmp =new StringIndexer().setInputCol(h).setOutputCol("label");
                rawData=indexTmp.fit(rawData).transform(rawData);

                headersRusult.add("label");

            }

            else {
                StringIndexer indexTmp=new StringIndexer().setInputCol(h).setOutputCol(h.toLowerCase()+"_cat");
                rawData=indexTmp.fit(rawData).transform(rawData);
                headersRusult.add(h.toLowerCase()+"_cat");

            }

        }

        String[] colList = headersRusult.toArray(new String[headersRusult.size()]);

        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(colList).setOutputCol("features");

        Dataset<Row> transformData = vectorAssembler.transform(rawData);

        Dataset<Row> finalData = transformData.select("label", "features");

        Dataset<Row>[] datasets = finalData.randomSplit(new double[]{0.7, 0.3});

        Dataset<Row> trainData = datasets[0];
        Dataset<Row> testData = datasets[1];

        NaiveBayes nb = new NaiveBayes();
        nb.setSmoothing(1);
        NaiveBayesModel model=nb.fit(trainData);
        Dataset<Row> prediction = model.transform(testData);

        prediction.show();


        MulticlassClassificationEvaluator evaluator= new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");


        double evaluate = evaluator.evaluate(prediction);

        System.out.println("Accuracy = " + evaluate);


    }

}
