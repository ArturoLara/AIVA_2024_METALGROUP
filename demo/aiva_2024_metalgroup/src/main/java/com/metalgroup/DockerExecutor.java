package com.metalgroup;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.awt.Rectangle;

public class DockerExecutor {
    private static final Pattern COORD_PATTERN = Pattern.compile("\\((\\d+),(\\d+),(\\d+),(\\d+)\\)");

    public static List<Rectangle> processImage(String imagePath) throws IOException, InterruptedException {

        String imageName = new File(imagePath).getName();
        String datasetPath = new File("dataset").getAbsolutePath();

        String[] command = {
                "docker", "run",
                "-e", "CONFIG=/App/config.json",
                "-e", "IMAGE=/App/dataset/" + imageName,
                "-v", datasetPath + ":/App",
                "artzulm/aiva_2024_metalgroup:1.0"
        };


        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        Process process = pb.start();

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line);
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IOException("Error en Docker: Código " + exitCode);
        }

        return parseDockerOutput(output.toString());
    }

    private static List<Rectangle> parseDockerOutput(String output) {
        List<Rectangle> rectangles = new ArrayList<>();
        Matcher matcher = COORD_PATTERN.matcher(output);

        while (matcher.find()) {
            int x = Integer.parseInt(matcher.group(1));
            int y = Integer.parseInt(matcher.group(2));
            int width = Integer.parseInt(matcher.group(3));
            int height = Integer.parseInt(matcher.group(4));

            if (x == 0 && y == 0 && width == 0 && height == 0) continue;

            rectangles.add(new Rectangle(x, y, width, height));
        }

        return rectangles.isEmpty() ? new ArrayList<>() : rectangles;
    }
}
