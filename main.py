import sys
from metal import preprocessing
from metal import defect_detector as detector


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]

    processed_image = preprocessing.process_image(image_path)
    defects = detector.detect_defects(processed_image)

    print("Processed image shape:", processed_image.shape)
    print("Detected defects:", defects)


if __name__ == "__main__":
    main()
