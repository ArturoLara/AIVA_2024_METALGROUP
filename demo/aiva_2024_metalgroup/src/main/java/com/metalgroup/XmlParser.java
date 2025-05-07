package com.metalgroup;

import org.w3c.dom.*;
import javax.xml.parsers.*;
import java.awt.Rectangle;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class XmlParser {
    public static List<Rectangle> parseGroundTruth(File xmlFile) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document doc = builder.parse(xmlFile);
        doc.getDocumentElement().normalize();

        List<Rectangle> boxes = new ArrayList<>();
        NodeList bndboxes = doc.getElementsByTagName("bndbox");

        for (int i = 0; i < bndboxes.getLength(); i++) {
            Element box = (Element) bndboxes.item(i);
            int xmin = getIntValue(box, "xmin");
            int ymin = getIntValue(box, "ymin");
            int xmax = getIntValue(box, "xmax");
            int ymax = getIntValue(box, "ymax");

            boxes.add(new Rectangle(xmin, ymin, xmax - xmin, ymax - ymin));
        }
        return boxes;
    }

    private static int getIntValue(Element parent, String tagName) {
        return Integer.parseInt(parent.getElementsByTagName(tagName)
                .item(0).getTextContent().trim());
    }
}
