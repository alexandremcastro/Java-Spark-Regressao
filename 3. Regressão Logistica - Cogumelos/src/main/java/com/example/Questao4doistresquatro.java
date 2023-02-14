package com.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;

public class Questao4doistresquatro {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("Mushrooms")
                .getOrCreate();

        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.ERROR);

        String FilePath = new String("hdfs://localhost:9000/user/castro/mushrooms.csv");

        Dataset<Row> data = spark.read()
                .option("header", true)
                .option("inferschema", true)
                .format("csv")
                .option("delimiter", ",")
                .load(FilePath);

        data.show();
        data.printSchema();
        long entradas = data.count();
        System.out.println("temos" + entradas + "em nosso dataset");

        String[] colnames = data.columns();

        String myRow = data.head().toString();
        for (int ind = 0; ind < colnames.length; ind++) {
            System.out.println(">>" + colnames[ind]);
            System.out.println(myRow.split(",")[ind]);
        }

        Dataset<Row> df = (data
                .select(col("class").as("label"), col("cap-shape"), col("cap-surface"),
                        col("cap-color"),
                        col("bruises"), col("odor"), col("gill-attachment"),
                        col("gill-spacing"), col("gill-size"),
                        col("gill-color"), col("stalk-shape"), col("stalk-root"),
                        col("stalk-surface-above-ring"),
                        col("stalk-surface-below-ring"), col("stalk-color-above-ring"),
                        col("stalk-color-below-ring"),
                        col("veil-type"), col("veil-color"), col("ring-number"),
                        col("ring-type"),
                        col("spore-print-color"), col("population"), col("habitat")));

        df.show(10);
        System.out.println("\n \n \n" + df.count() + "\n \n \n");

        Dataset<Row> df_atualizado = df.na().drop();

        StringIndexer capshapeIndexer = new StringIndexer().setInputCol("cap-shape").setOutputCol("capshapeIndex");
        StringIndexer capsurfaceIndexer = new StringIndexer().setInputCol("cap-surface")
                .setOutputCol("capsurfaceIndex");
        StringIndexer capcolorIndexer = new StringIndexer().setInputCol("cap-color").setOutputCol("capcolorIndex");
        StringIndexer bruisesIndexer = new StringIndexer().setInputCol("bruises").setOutputCol("bruisesIndex");
        StringIndexer odorIndexer = new StringIndexer().setInputCol("odor").setOutputCol("odorIndex");
        StringIndexer gillattachmentIndexer = new StringIndexer().setInputCol("gill-attachment")
                .setOutputCol("gillattachmentIndex");
        StringIndexer gillspacingIndexer = new StringIndexer().setInputCol("gill-spacing")
                .setOutputCol("gillspacingIndex");
        StringIndexer gillsizeIndexer = new StringIndexer().setInputCol("gill-size").setOutputCol("gillsizeIndex");
        StringIndexer gillcolorIndexer = new StringIndexer().setInputCol("gill-color").setOutputCol("gillcolorIndex");
        StringIndexer stalkshapeIndexer = new StringIndexer().setInputCol("stalk-shape")
                .setOutputCol("stalkshapeIndex");
        StringIndexer stalkrootIndexer = new StringIndexer().setInputCol("stalk-root").setOutputCol("stalkrootIndex");
        StringIndexer stalksurfaceaboveringIndexer = new StringIndexer().setInputCol("stalk-surface-above-ring")
                .setOutputCol("stalksurfaceaboveringIndex");
        StringIndexer stalksurfacebelowringIndexer = new StringIndexer().setInputCol("stalk-surface-below-ring")
                .setOutputCol("stalksurfacebelowringIndex");
        StringIndexer stalkcoloraboveringIndexer = new StringIndexer().setInputCol("stalk-color-above-ring")
                .setOutputCol("stalkcoloraboveringIndex");
        StringIndexer stalkcolorbelowringIndexer = new StringIndexer().setInputCol("stalk-color-below-ring")
                .setOutputCol("stalkcolorbelowringIndex");
        StringIndexer veiltypeIndexer = new StringIndexer().setInputCol("veil-type").setOutputCol("veiltypeIndex");
        StringIndexer veilcolorIndexer = new StringIndexer().setInputCol("veil-color").setOutputCol("veilcolorIndex");
        StringIndexer ringnumberIndexer = new StringIndexer().setInputCol("ring-number")
                .setOutputCol("ringnumberIndex");
        StringIndexer ringtypeIndexer = new StringIndexer().setInputCol("ring-type").setOutputCol("ringtypeIndex");
        StringIndexer sporeprintcolorIndexer = new StringIndexer().setInputCol("spore-print-color")
                .setOutputCol("sporeprintcolorIndex");
        StringIndexer populationIndexer = new StringIndexer().setInputCol("population").setOutputCol("populationIndex");
        StringIndexer habitatIndexer = new StringIndexer().setInputCol("habitat").setOutputCol("habitatIndex");
        StringIndexer labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("labelIndex");

        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[] {
                        "capshapeIndex",
                        "capsurfaceIndex",
                        "capcolorIndex",
                        "bruisesIndex",
                        "odorIndex",
                        "gillattachmentIndex",
                        "gillspacingIndex",
                        "gillsizeIndex",
                        "gillcolorIndex",
                        "stalkshapeIndex",
                        "stalkrootIndex",
                        "stalksurfaceaboveringIndex",
                        "stalksurfacebelowringIndex",
                        "stalkcoloraboveringIndex",
                        "stalkcolorbelowringIndex",
                        "veiltypeIndex",
                        "veilcolorIndex",
                        "ringnumberIndex",
                        "ringtypeIndex",
                        "sporeprintcolorIndex",
                        "populationIndex",
                        "habitatIndex" })
                .setOutputCol("features"));

        Dataset<Row>[] dadosTreinamentoTeste = df_atualizado.randomSplit(new double[] { 0.7, 0.3 });
        Dataset<Row> dadosTreinamento = dadosTreinamentoTeste[0];
        Dataset<Row> dadosTeste = dadosTreinamentoTeste[1];

        // Regressão Logistica
        LogisticRegression lr = new LogisticRegression().setLabelCol("labelIndex").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
                capshapeIndexer,
                capsurfaceIndexer,
                capcolorIndexer,
                bruisesIndexer,
                odorIndexer,
                gillattachmentIndexer,
                gillspacingIndexer,
                gillsizeIndexer,
                gillcolorIndexer,
                stalkshapeIndexer,
                stalkrootIndexer,
                stalksurfaceaboveringIndexer,
                stalksurfacebelowringIndexer,
                stalkcoloraboveringIndexer,
                stalkcolorbelowringIndexer,
                veiltypeIndexer,
                veilcolorIndexer,
                ringnumberIndexer,
                ringtypeIndexer,
                sporeprintcolorIndexer,
                populationIndexer,
                habitatIndexer,
                labelIndexer,
                assembler,
                lr });

        PipelineModel model = pipeline.fit(dadosTreinamento);
        Dataset<Row> predictions = model.transform(dadosTeste);

        PCA pcaModel = (new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(5));

        LogisticRegression lrPca = new LogisticRegression().setLabelCol("labelIndex").setFeaturesCol("pcaFeatures");
        Pipeline pipelinePca = new Pipeline().setStages(new PipelineStage[] {
                capshapeIndexer,
                capsurfaceIndexer,
                capcolorIndexer,
                bruisesIndexer,
                odorIndexer,
                gillattachmentIndexer,
                gillspacingIndexer,
                gillsizeIndexer,
                gillcolorIndexer,
                stalkshapeIndexer,
                stalkrootIndexer,
                stalksurfaceaboveringIndexer,
                stalksurfacebelowringIndexer,
                stalkcoloraboveringIndexer,
                stalkcolorbelowringIndexer,
                veiltypeIndexer,
                veilcolorIndexer,
                ringnumberIndexer,
                ringtypeIndexer,
                sporeprintcolorIndexer,
                populationIndexer,
                habitatIndexer,
                labelIndexer,
                assembler,
                pcaModel,
                lrPca });

        PipelineModel modelPca = pipelinePca.fit(dadosTreinamento);
        Dataset<Row> predictionsPCA = modelPca.transform(dadosTeste);

        // Resultado sem PCA
        predictions.select("labelIndex", "prediction", "features").show(10);

        // Matriz de confusão sem PCA
        System.out.println("Matriz de confusão Regressão Logistica sem PCA");
        predictions.groupBy(col("labelIndex"), col("prediction")).count().show();

        // Accuracy sem PCA
        MulticlassClassificationEvaluator lr_Evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("labelIndex")
                .setPredictionCol("prediction").setMetricName("accuracy");
        double lr_accuracy = lr_Evaluator.evaluate(predictions);
        System.out.println("\n \n \n Acuracia Regressão Logística com PCA: " + lr_accuracy * 100 + "% \n \n \n");

        // Resultado com PCA
        predictionsPCA.select(col("labelIndex"), col("prediction"), col("pcaFeatures"), col("pcaFeatures"),
                col("rawPrediction"), col("probability")).show(5);

        // Matriz de confusão com PCA
        System.out.println("Matriz de confusão Regressão Logistica com PCA");
        predictionsPCA.groupBy(col("labelIndex"), col("prediction")).count().show();

        // Accurracy com PCA
        MulticlassClassificationEvaluator lr_EvaluatorPca = new MulticlassClassificationEvaluator()
                .setLabelCol("labelIndex").setPredictionCol("prediction").setMetricName("accuracy");
        double lr_accuracyPca = lr_EvaluatorPca.evaluate(predictionsPCA);
        System.out.println("\n \n \n Acuracia Regressão Logística com PCA: " + lr_accuracyPca * 100 + "% \n \n \n");

        // Random Forest
        RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("labelIndex").setFeaturesCol("features");
        Pipeline pipelineRF = new Pipeline().setStages(new PipelineStage[] {
                capshapeIndexer,
                capsurfaceIndexer,
                capcolorIndexer,
                bruisesIndexer,
                odorIndexer,
                gillattachmentIndexer,
                gillspacingIndexer,
                gillsizeIndexer,
                gillcolorIndexer,
                stalkshapeIndexer,
                stalkrootIndexer,
                stalksurfaceaboveringIndexer,
                stalksurfacebelowringIndexer,
                stalkcoloraboveringIndexer,
                stalkcolorbelowringIndexer,
                veiltypeIndexer,
                veilcolorIndexer,
                ringnumberIndexer,
                ringtypeIndexer,
                sporeprintcolorIndexer,
                populationIndexer,
                habitatIndexer,
                labelIndexer,
                assembler,
                rf });

        PipelineModel rf_model = pipelineRF.fit(dadosTreinamento);
        Dataset<Row> rf_predictions = rf_model.transform(dadosTeste);
        rf_predictions.show();

        // Resultado RandomForest SEM PCA
        rf_predictions.select("labelIndex", "prediction", "features").show(5);

        // Matriz de confusão RandomForest sem PCA
        System.out.println("Matriz de confusão RandomForest sem PCA");
        rf_predictions.groupBy(col("labelIndex"), col("prediction")).count().show();

        // Accuracy RandomForest sem PCA
        MulticlassClassificationEvaluator rf_Evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("labelIndex")
                .setPredictionCol("prediction").setMetricName("accuracy");
        double rf_accuracy = rf_Evaluator.evaluate(rf_predictions);
        System.out.println("\n \n \n Acuracia RandomForest sem PCA: " + rf_accuracy * 100 + "% \n \n \n");

        RandomForestClassifier rfPca = new RandomForestClassifier().setLabelCol("labelIndex")
                .setFeaturesCol("pcaFeatures");
        Pipeline rf_pipelinePca = new Pipeline().setStages(new PipelineStage[] {
                capshapeIndexer,
                capsurfaceIndexer,
                capcolorIndexer,
                bruisesIndexer,
                odorIndexer,
                gillattachmentIndexer,
                gillspacingIndexer,
                gillsizeIndexer,
                gillcolorIndexer,
                stalkshapeIndexer,
                stalkrootIndexer,
                stalksurfaceaboveringIndexer,
                stalksurfacebelowringIndexer,
                stalkcoloraboveringIndexer,
                stalkcolorbelowringIndexer,
                veiltypeIndexer,
                veilcolorIndexer,
                ringnumberIndexer,
                ringtypeIndexer,
                sporeprintcolorIndexer,
                populationIndexer,
                habitatIndexer,
                labelIndexer,
                assembler,
                pcaModel,
                rfPca });

        PipelineModel rf_modelPca = rf_pipelinePca.fit(dadosTreinamento);
        Dataset<Row> rf_predictionsPca = rf_modelPca.transform(dadosTeste);
        rf_predictionsPca.show(7);

        // Resultado RandomForest com PCA
        rf_predictionsPca.select("labelIndex", "prediction", "features").show(5);

        // 'Matriz de confusão' RandomForest com PCA
        System.out.println("Matriz de confusão RandomForest com PCA");
        rf_predictionsPca.groupBy(col("labelIndex"), col("prediction")).count().show();

        // Accuracy RandomForest com PCA
        MulticlassClassificationEvaluator rf_EvaluatorPca = new MulticlassClassificationEvaluator()
                .setLabelCol("labelIndex")
                .setPredictionCol("prediction").setMetricName("accuracy");
        double rf_accuracyPca = rf_EvaluatorPca.evaluate(rf_predictionsPca);
        System.out.println("\n \n \n Acuracia RandomForest sem PCA: " + rf_accuracyPca * 100 + "% \n \n \n");

        // Todas as accuracys
        System.out.println("\n  Acuracia Regressão Logística sem PCA: " + lr_accuracy * 100 + "% ");
        System.out.println("  Acuracia Regressão Logística com PCA: " + lr_accuracyPca * 100 + "% ");
        System.out.println("  Acuracia RandomForest sem PCA: " + rf_accuracy * 100 + "% ");
        System.out.println("  Acuracia RandomForest com PCA: " + rf_accuracyPca * 100 + "% \n");

    }
}
