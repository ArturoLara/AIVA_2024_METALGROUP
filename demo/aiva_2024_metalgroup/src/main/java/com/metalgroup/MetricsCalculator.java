// MetricsCalculator.java
package com.metalgroup;

import java.awt.Rectangle;
import java.util.*;

public class MetricsCalculator {
    public static double calculateIoU(Rectangle a, Rectangle b) {
        int interLeft = Math.max(a.x, b.x);
        int interTop = Math.max(a.y, b.y);
        int interRight = Math.min(a.x + a.width, b.x + b.width);
        int interBottom = Math.min(a.y + a.height, b.y + b.height);

        if (interRight < interLeft || interBottom < interTop) return 0;

        double interArea = (interRight - interLeft) * (interBottom - interTop);
        double unionArea = (a.width * a.height) + (b.width * b.height) - interArea;

        return interArea / unionArea;
    }

    public static Map<String, Integer> calculateDetectionMetrics(List<Rectangle> groundTruth,
                                                                 List<Rectangle> predictions,
                                                                 double iouThreshold) {
        Map<String, Integer> metrics = new HashMap<>();
        metrics.put("TP", 0);
        metrics.put("FP", 0);
        metrics.put("FN", 0);

        Set<Integer> matchedGt = new HashSet<>();
        for (Rectangle pred : predictions) {
            boolean matched = false;
            for (int i = 0; i < groundTruth.size(); i++) {
                if (!matchedGt.contains(i) && calculateIoU(pred, groundTruth.get(i)) >= iouThreshold) {
                    matchedGt.add(i);
                    metrics.put("TP", metrics.get("TP") + 1);
                    matched = true;
                    break;
                }
            }
            if (!matched) metrics.put("FP", metrics.get("FP") + 1);
        }
        metrics.put("FN", groundTruth.size() - matchedGt.size());
        return metrics;
    }

    public static List<Double> calculateAllIoUs(List<Rectangle> groundTruth, List<Rectangle> predictions) {
        List<Double> ious = new ArrayList<>();
        for (Rectangle gt : groundTruth) {
            for (Rectangle pred : predictions) {
                ious.add(calculateIoU(gt, pred));
            }
        }
        return ious;
    }

    public static double calculateAverageIoU(List<Double> ious) {
        return ious.stream().mapToDouble(d -> d).average().orElse(0);
    }

    public static Map<String, Double> calculateGlobalMetrics(List<DetectionResult> results) {
        Map<String, Double> global = new HashMap<>();
        global.put("TotalTP", 0.0);
        global.put("TotalFP", 0.0);
        global.put("TotalFN", 0.0);
        global.put("TotalTime", 0.0);
        global.put("AvgIoU", 0.0);

        List<Double> allIous = new ArrayList<>();
        for (DetectionResult result : results) {
            global.put("TotalTP", global.get("TotalTP") + result.getDetectionMetrics().get("TP"));
            global.put("TotalFP", global.get("TotalFP") + result.getDetectionMetrics().get("FP"));
            global.put("TotalFN", global.get("TotalFN") + result.getDetectionMetrics().get("FN"));
            global.put("TotalTime", global.get("TotalTime") + result.getProcessingTime());
            allIous.addAll(result.getIouScores());
        }

        global.put("AvgIoU", calculateAverageIoU(allIous));
        global.put("TotalTime", global.get("TotalTime") / 1000); // Convertir a segundos

        return global;
    }
}
