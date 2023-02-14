package com.example;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class TP9 {
        public static void main(String[] args) {
                SparkSession spark = SparkSession
                                .builder()
                                .master("local[*]")
                                .appName("tp9")
                                .getOrCreate();

                Dataset<Row> dataset = spark.read()
                                .option("header", true)
                                .option("inferschema", true)
                                .format("csv")
                                .load("hdfs://localhost:9000/user/castro/vgsales.csv");

                // Rank Name Platform Year Genre Publisher NA_Sales EU_Sales JP_Sales
                // Other_Sales Global_Sales

                Dataset<Row> df = (dataset
                                .select(col("Global_Sales").as("label"), col("Platform"), col("Year"), col("Genre"),
                                                col("Publisher"),
                                                col("NA_Sales"), col("EU_Sales"), col("JP_Sales"), col("Other_Sales")));

                StringIndexer platformIndexer = new StringIndexer();
                platformIndexer.setInputCol("Platform");
                platformIndexer.setOutputCol("PlatformIndex");

                StringIndexer genreIndexer = new StringIndexer();
                genreIndexer.setInputCol("Genre");
                genreIndexer.setOutputCol("GenreIndex");

                StringIndexer publisherIndexer = new StringIndexer();
                publisherIndexer.setInputCol("Publisher");
                publisherIndexer.setOutputCol("PublisherIndex");

                df = platformIndexer.fit(df).transform(df);
                df = genreIndexer.fit(df).transform(df);
                df = publisherIndexer.fit(df).transform(df);

                VectorAssembler assembler = (new VectorAssembler()
                                .setInputCols(new String[] {
                                                "PlatformIndex",
                                                "Year",
                                                "GenreIndex",
                                                "PublisherIndex",
                                                "NA_Sales",
                                                "EU_Sales",
                                                "Other_Sales" })
                                .setOutputCol("features"));

                Dataset<Row> predicao = assembler.transform(df).select("label", "features");
                predicao.show();

                LinearRegression lr = new LinearRegression();
                LinearRegressionModel lrModel = lr.fit(predicao);
                lrModel.transform(predicao).show();

                Dataset<Row>[] dadosTreinamentoTeste = predicao.randomSplit(new double[] { 0.8, 0.2 });
                Dataset<Row> dadosTreinamento = dadosTreinamentoTeste[0];
                Dataset<Row> dadosTeste = dadosTreinamentoTeste[1];

                LinearRegressionModel lrModel2 = lr.fit(dadosTreinamento);
                lrModel2.transform(dadosTeste).show();

                System.out.println("R2: " + lrModel2.summary().r2()); // QUANTO MAIS PRÓXIMO DE 1 MELHOR
                System.out.println("Erro quadrático médio: " + lrModel2.summary().rootMeanSquaredError()); // QUANTO
                System.out.println("\n \n \n"); // MENOR
                // MELHOR
        }
}
