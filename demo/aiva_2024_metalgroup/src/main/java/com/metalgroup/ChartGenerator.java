// ChartGenerator.java
package com.metalgroup;

import org.jfree.chart.*;
import org.jfree.chart.plot.*;
import org.jfree.data.category.*;
import org.jfree.data.statistics.*;
import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;
import org.jfree.chart.renderer.category.BarRenderer;

public class ChartGenerator {
    public static void generateAllCharts(List<DetectionResult> results, Map<String, Double> globalMetrics) {
        try {
            createIOUDistributionChart(results, "iou_distribution.png");
            createPerformanceChart(results, "performance.png");
            createConfusionMatrixChart(globalMetrics, "confusion_matrix.png");
        } catch (Exception e) {
            System.err.println("Error generando gráficos: " + e.getMessage());
        }
    }

    private static void createIOUDistributionChart(List<DetectionResult> results, String filename) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        results.stream()
                .flatMap(r -> r.getIouScores().stream())
                .forEach(iou -> dataset.addValue(1, "IOU", String.format("%.2f", iou)));

        JFreeChart chart = ChartFactory.createBarChart(
                "Distribución de IoU", "Rango IoU", "Frecuencia", dataset);

        chart.setBackgroundPaint(Color.white);
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.lightGray);

        ChartUtils.saveChartAsPNG(new File(filename), chart, 800, 600);
    }

    private static void createPerformanceChart(List<DetectionResult> results, String filename) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        results.forEach(r -> dataset.addValue(r.getProcessingTime(), "Tiempo", r.getImageName()));

        JFreeChart chart = ChartFactory.createLineChart(
                "Tiempo de Procesamiento por Imagen",
                "Imagen",
                "Tiempo (ms)",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        ChartUtils.saveChartAsPNG(new File(filename), chart, 1200, 600);
    }

    private static void createConfusionMatrixChart(Map<String, Double> metrics, String filename) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.addValue(metrics.get("TotalTP"), "Verdaderos", "Positivos");
        dataset.addValue(metrics.get("TotalFP"), "Falsos", "Positivos");
        dataset.addValue(metrics.get("TotalFN"), "Falsos", "Negativos");

        JFreeChart chart = ChartFactory.createBarChart(
                "Matriz de Confusión",
                "Categoría",
                "Cantidad",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        BarRenderer renderer = (BarRenderer) chart.getCategoryPlot().getRenderer();
        renderer.setSeriesPaint(0, Color.GREEN);
        renderer.setSeriesPaint(1, Color.RED);
        renderer.setSeriesPaint(2, Color.ORANGE);

        ChartUtils.saveChartAsPNG(new File(filename), chart, 800, 600);
    }
}
