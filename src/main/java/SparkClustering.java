import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeoutException;

public class SparkClustering {
    public static void main(String[] args) throws StreamingQueryException, TimeoutException {
        // Создание SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("Spark Clustering Example")
                .config("spark.master", "local")
                .getOrCreate();

        // Источник данных: файл
        DataSource fileDataSource = new FileDataSource("C:\\Users\\Elena\\Desktop\\FOLDER\\.csv", spark);
        Dataset<Row> fileData = fileDataSource.getData();
        MetaInf fileMetaInf = fileDataSource.getMetaInf();

        // Применение алгоритмов кластеризации для файловых данных
        Klasterisator fileKMeansClusterisator = new KMeansClusterisator();
        fileKMeansClusterisator.getClusters(fileData, fileMetaInf);


/*

        // Источник данных: Kafka
        DataSource kafkaDataSource = new KafkaDataSource("kafkaTopic", spark);
        Dataset<Row> kafkaData = kafkaDataSource.getData();
        MetaInf kafkaMetaInf = kafkaDataSource.getMetaInf();

        // Применение алгоритмов кластеризации для данных из Kafka
        Klasterisator kafkaKMeansClusterisator = new KMeansClusterisator();
        kafkaKMeansClusterisator.getClusters(kafkaData, kafkaMetaInf);

        // Обработка данных из Kafka и вывод в консоль
        StreamingQuery query = kafkaData
                .writeStream()
                .outputMode("append")
                .format("console")
                .start();

        query.awaitTermination();
*/


        // Закрытие SparkSession
        spark.stop();
    }
}

    interface DataSource {
        Dataset<Row> getData();
        MetaInf getMetaInf();
    }

    class FileDataSource implements DataSource {
        private final String filePath;
        private final SparkSession spark;

        FileDataSource(String filePath, SparkSession spark) {
            this.filePath = filePath;
            this.spark = spark;
        }

        @Override
        public Dataset<Row> getData() {
            // Чтение данных из файла и создание Dataset<Row> с автоматическим определением схемы
            return spark.read()
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .csv(filePath);
        }

        @Override
        public MetaInf getMetaInf() {
            Dataset<Row> data = getData();
            return new MetaInf(data.schema().size(), Arrays.asList(data.schema().fieldNames()));
        }
    }

    class KafkaDataSource implements DataSource {
        private final String topicName;
        private final SparkSession spark;

        KafkaDataSource(String topicName, SparkSession spark) {
            this.topicName = topicName;
            this.spark = spark;
        }

        @Override
        public Dataset<Row> getData() {
            // Чтение данных из Kafka и создание Dataset<Row>
            Dataset<Row> kafkaStream = spark.readStream()
                    .format("kafka")
                    .option("kafka.bootstrap.servers", "localhost:9092")
                    .option("subscribe", topicName)
                    .load()
                    .selectExpr("CAST(value AS STRING)")
                    .as(Encoders.STRING())
                    .toDF();

            return kafkaStream;
        }

        @Override
        public MetaInf getMetaInf() {
            Dataset<Row> data = getData();
            return new MetaInf(data.schema().size(), Arrays.asList(data.schema().fieldNames()));
        }
    }

    class MetaInf {
        private final int colCount;
        private final List<String> columns;

        MetaInf(int colCount, List<String> columns) {
            this.colCount = colCount;
            this.columns = columns;
        }

        String[] getColNames() {
            return columns.toArray(new String[0]);
        }

        int getColCount() {
            return colCount;
        }
    }

    interface Klasterisator {
        void getClusters(Dataset<Row> data, MetaInf metaInf);
    }

    class KMeansClusterisator implements Klasterisator {
        @Override
        public void getClusters(Dataset<Row> data, MetaInf metaInf) {
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(metaInf.getColNames())
                    .setOutputCol("features");

            Dataset<Row> featureData = assembler.transform(data);

            int k = 130; // Количество кластеров
            KMeans kmeans = new KMeans()
                    .setK(k)
                    .setSeed(1L)
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster");

            KMeansModel model = kmeans.fit(featureData);
            Dataset<Row> predictions = model.transform(featureData);

            System.out.println("Cluster Centers:");
            for (Vector center : model.clusterCenters()) {
                System.out.println(center);
            }

            System.out.println("Cluster Assignments:");
            predictions.select("features", "cluster").show();
        }
    }

