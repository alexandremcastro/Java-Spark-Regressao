package com.example;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class Questao4um {
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
                                .load("hdfs://localhost:9000/user/castro/mushrooms.csv");

                Dataset<Row> df = (dataset
                                .select(col("classe").as("label"), col("cap_shape"), col("cap_surface"),
                                                col("cap_color"),
                                                col("bruises"), col("odor"), col("gill_attachment"),
                                                col("gill_spacing"), col("gill_size"),
                                                col("gill_color"), col("stalk_shape"), col("stalk_root"),
                                                col("stalk_surface_above_ring"),
                                                col("stalk_surface_below_ring"), col("stalk_color_above_ring"),
                                                col("stalk_color_below_ring"),
                                                col("veil_type"), col("veil_color"), col("ring_number"),
                                                col("ring_type"),
                                                col("spore_print_color"), col("population"), col("habitat")));

                StringIndexer cap_shapeIndexer = new StringIndexer();
                cap_shapeIndexer.setInputCol("cap_shape");
                cap_shapeIndexer.setOutputCol("cap_shapeIndex");

                StringIndexer cap_surfaceIndexer = new StringIndexer();
                cap_surfaceIndexer.setInputCol("cap_surface");
                cap_surfaceIndexer.setOutputCol("cap_surfaceIndex");

                StringIndexer cap_colorIndexer = new StringIndexer();
                cap_colorIndexer.setInputCol("cap_color");
                cap_colorIndexer.setOutputCol("cap_colorIndex");

                StringIndexer bruisesIndexer = new StringIndexer();
                bruisesIndexer.setInputCol("bruises");
                bruisesIndexer.setOutputCol("bruisesIndex");

                StringIndexer odorIndexer = new StringIndexer();
                odorIndexer.setInputCol("odor");
                odorIndexer.setOutputCol("odorIndex");

                StringIndexer gill_attachmentIndexer = new StringIndexer();
                gill_attachmentIndexer.setInputCol("gill_attachment");
                gill_attachmentIndexer.setOutputCol("gill_attachmentIndex");

                StringIndexer gill_spacingIndexer = new StringIndexer();
                gill_spacingIndexer.setInputCol("gill_spacing");
                gill_spacingIndexer.setOutputCol("gill_spacingIndex");

                StringIndexer gill_sizeIndexer = new StringIndexer();
                gill_sizeIndexer.setInputCol("gill_size");
                gill_sizeIndexer.setOutputCol("gill_sizeIndex");

                StringIndexer gill_colorIndexer = new StringIndexer();
                gill_colorIndexer.setInputCol("gill_color");
                gill_colorIndexer.setOutputCol("gill_colorIndex");

                StringIndexer stalk_shapeIndexer = new StringIndexer();
                stalk_shapeIndexer.setInputCol("stalk_shape");
                stalk_shapeIndexer.setOutputCol("stalk_shapeIndex");

                StringIndexer stalk_rootIndexer = new StringIndexer();
                stalk_rootIndexer.setInputCol("stalk_root");
                stalk_rootIndexer.setOutputCol("stalk_rootIndex");

                StringIndexer stalk_surface_above_ringIndexer = new StringIndexer();
                stalk_surface_above_ringIndexer.setInputCol("stalk_surface_above_ring");
                stalk_surface_above_ringIndexer.setOutputCol("stalk_surface_above_ringIndex");

                StringIndexer stalk_surface_below_ringIndexer = new StringIndexer();
                stalk_surface_below_ringIndexer.setInputCol("stalk_surface_below_ring");
                stalk_surface_below_ringIndexer.setOutputCol("stalk_surface_below_ringIndex");

                StringIndexer stalk_color_above_ringIndexer = new StringIndexer();
                stalk_color_above_ringIndexer.setInputCol("stalk_color_above_ring");
                stalk_color_above_ringIndexer.setOutputCol("stalk_color_above_ringIndex");

                StringIndexer stalk_color_below_ringIndexer = new StringIndexer();
                stalk_color_below_ringIndexer.setInputCol("stalk_color_below_ring");
                stalk_color_below_ringIndexer.setOutputCol("stalk_color_below_ringIndex");

                StringIndexer veil_typeIndexer = new StringIndexer();
                veil_typeIndexer.setInputCol("veil_type");
                veil_typeIndexer.setOutputCol("veil_typeIndex");

                StringIndexer veil_colorIndexer = new StringIndexer();
                veil_colorIndexer.setInputCol("veil_color");
                veil_colorIndexer.setOutputCol("veil_colorIndex");

                StringIndexer ring_numberIndexer = new StringIndexer();
                ring_numberIndexer.setInputCol("ring_number");
                ring_numberIndexer.setOutputCol("ring_numberIndex");

                StringIndexer ring_typeIndexer = new StringIndexer();
                ring_typeIndexer.setInputCol("ring_type");
                ring_typeIndexer.setOutputCol("ring_typeIndex");

                StringIndexer spore_print_colorIndexer = new StringIndexer();
                spore_print_colorIndexer.setInputCol("spore_print_color");
                spore_print_colorIndexer.setOutputCol("spore_print_colorIndex");

                StringIndexer populationIndexer = new StringIndexer();
                populationIndexer.setInputCol("population");
                populationIndexer.setOutputCol("populationIndex");

                StringIndexer habitatIndexer = new StringIndexer();
                habitatIndexer.setInputCol("habitat");
                habitatIndexer.setOutputCol("habitatIndex");

                df = cap_shapeIndexer.fit(df).transform(df);
                df = cap_surfaceIndexer.fit(df).transform(df);
                df = cap_colorIndexer.fit(df).transform(df);
                df = bruisesIndexer.fit(df).transform(df);
                df = odorIndexer.fit(df).transform(df);
                df = gill_attachmentIndexer.fit(df).transform(df);
                df = gill_spacingIndexer.fit(df).transform(df);
                df = gill_sizeIndexer.fit(df).transform(df);
                df = gill_colorIndexer.fit(df).transform(df);
                df = stalk_shapeIndexer.fit(df).transform(df);
                df = stalk_rootIndexer.fit(df).transform(df);
                df = stalk_surface_above_ringIndexer.fit(df).transform(df);
                df = stalk_surface_below_ringIndexer.fit(df).transform(df);
                df = stalk_color_above_ringIndexer.fit(df).transform(df);
                df = stalk_color_below_ringIndexer.fit(df).transform(df);
                df = veil_typeIndexer.fit(df).transform(df);
                df = veil_colorIndexer.fit(df).transform(df);
                df = ring_numberIndexer.fit(df).transform(df);
                df = ring_typeIndexer.fit(df).transform(df);
                df = spore_print_colorIndexer.fit(df).transform(df);
                df = populationIndexer.fit(df).transform(df);
                df = habitatIndexer.fit(df).transform(df);

                df.show();

                VectorAssembler assembler = (new VectorAssembler()
                                .setInputCols(new String[] {
                                                "cap_shapeIndex",
                                                "cap_surfaceIndex",
                                                "cap_colorIndex",
                                                "bruisesIndex",
                                                "odorIndex",
                                                "gill_attachmentIndex",
                                                "gill_spacingIndex",
                                                "gill_sizeIndex",
                                                "gill_colorIndex",
                                                "stalk_shapeIndex",
                                                "stalk_rootIndex",
                                                "stalk_surface_above_ringIndex",
                                                "stalk_surface_below_ringIndex",
                                                "stalk_color_above_ringIndex",
                                                "stalk_color_below_ringIndex",
                                                "veil_typeIndex",
                                                "veil_colorIndex",
                                                "ring_numberIndex",
                                                "ring_typeIndex",
                                                "spore_print_colorIndex",
                                                "populationIndex" })
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

                System.out.println("R2: " + lrModel2.summary().r2());

                System.out.println("Erro quadrático médio: " + lrModel2.summary().rootMeanSquaredError());
                System.out.println("\n \n \n");
        }
}
