// DetectionResult.java
package com.metalgroup;

import java.awt.Rectangle;
import java.util.List;
import java.util.Map;

public class DetectionResult {
    private String imageName;
    private List<Rectangle> groundTruth;
    private List<Rectangle> predictions;
    private double processingTime;
    private List<Double> iouScores;
    private Map<String, Integer> detectionMetrics;

    public DetectionResult(String imageName) {
        this.imageName = imageName;
    }

    // Getters y setters
    public String getImageName() { return imageName; }
    public List<Rectangle> getGroundTruth() { return groundTruth; }
    public void setGroundTruth(List<Rectangle> groundTruth) { this.groundTruth = groundTruth; }
    public List<Rectangle> getPredictions() { return predictions; }
    public void setPredictions(List<Rectangle> predictions) { this.predictions = predictions; }
    public double getProcessingTime() { return processingTime; }
    public void setProcessingTime(double processingTime) { this.processingTime = processingTime; }
    public List<Double> getIouScores() { return iouScores; }
    public void setIouScores(List<Double> iouScores) { this.iouScores = iouScores; }
    public Map<String, Integer> getDetectionMetrics() { return detectionMetrics; }
    public void setDetectionMetrics(Map<String, Integer> detectionMetrics) { this.detectionMetrics = detectionMetrics; }
}
