// Main.java
package com.metalgroup;

import java.awt.Rectangle;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;
import org.apache.commons.csv.*;

public class Main {
    private static final double IOU_THRESHOLD = 0.5;

    public static void main(String[] args) throws Exception {
        List<DetectionResult> results = new ArrayList<>();
        File datasetDir = new File("dataset");

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Future<DetectionResult>> futures = new ArrayList<>();

        for (File imageFile : getImageFiles(datasetDir)) {
            futures.add(executor.submit(() -> processImage(imageFile)));
        }

        for (Future<DetectionResult> future : futures) {
            results.add(future.get());
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        Map<String, Double> globalMetrics = MetricsCalculator.calculateGlobalMetrics(results);
        generateReports(results, globalMetrics);
        saveMetricsToCSV(results, globalMetrics);

        System.out.println("Proceso completado. Resultados guardados en:");
        System.out.println("- metrics_report.csv");
        System.out.println("- iou_distribution.png");
        System.out.println("- performance.png");
        System.out.println("- confusion_matrix.png");
    }

    private static List<File> getImageFiles(File datasetDir) {
        return Arrays.stream(Objects.requireNonNull(datasetDir.listFiles()))
                .filter(f -> f.getName().toLowerCase().endsWith(".jpg"))
                .collect(Collectors.toList());
    }

    private static DetectionResult processImage(File imageFile) {
        DetectionResult result = new DetectionResult(imageFile.getName());
        long startTime = System.nanoTime();

        try {
            List<Rectangle> predictions = DockerExecutor.processImage(imageFile.getAbsolutePath());
            File xmlFile = new File(imageFile.getAbsolutePath().replace(".jpg", ".xml"));
            List<Rectangle> groundTruth = XmlParser.parseGroundTruth(xmlFile);

            result.setGroundTruth(groundTruth);
            result.setPredictions(predictions);
            result.setProcessingTime((System.nanoTime() - startTime) / 1e6);
            result.setIouScores(MetricsCalculator.calculateAllIoUs(groundTruth, predictions));
            result.setDetectionMetrics(MetricsCalculator.calculateDetectionMetrics(
                    groundTruth, predictions, IOU_THRESHOLD));

        } catch (Exception e) {
            System.err.println("Error procesando " + imageFile.getName() + ": " + e.getMessage());
        }
        return result;
    }

    private static void generateReports(List<DetectionResult> results, Map<String, Double> globalMetrics) {
        ChartGenerator.generateAllCharts(results, globalMetrics);
    }

    private static void saveMetricsToCSV(List<DetectionResult> results, Map<String, Double> globalMetrics) {
        try (Writer writer = Files.newBufferedWriter(Paths.get("metrics_report.csv"))) {
            CSVPrinter csvPrinter = CSVFormat.DEFAULT.withHeader(
                    "Imagen",
                    "TP",
                    "FP",
                    "FN",
                    "Tiempo(ms)",
                    "AvgIoU"
            ).print(writer);

            for (DetectionResult result : results) {
                csvPrinter.printRecord(
                        result.getImageName(),
                        result.getDetectionMetrics().get("TP"),
                        result.getDetectionMetrics().get("FP"),
                        result.getDetectionMetrics().get("FN"),
                        String.format("%.2f", result.getProcessingTime()),
                        String.format("%.2f", MetricsCalculator.calculateAverageIoU(result.getIouScores()))
                );
            }

            csvPrinter.println();
            csvPrinter.printRecord("Métricas Globales");
            csvPrinter.printRecord("Total TP", globalMetrics.get("TotalTP"));
            csvPrinter.printRecord("Total FP", globalMetrics.get("TotalFP"));
            csvPrinter.printRecord("Total FN", globalMetrics.get("TotalFN"));
            csvPrinter.printRecord("Tiempo Total (s)", String.format("%.2f", globalMetrics.get("TotalTime")));
            csvPrinter.printRecord("IoU Promedio", String.format("%.2f", globalMetrics.get("AvgIoU")));

            csvPrinter.flush();
        } catch (IOException e) {
            System.err.println("Error escribiendo CSV: " + e.getMessage());
        }
    }
}
