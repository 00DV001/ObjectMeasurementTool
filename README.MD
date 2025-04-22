# Object Detection and Dimension Measurement using OpenCV

This project performs object detection and real-world dimension measurement using OpenCV. An A4 sheet is used as a reference object, and any item placed on top of it is detected, outlined, and measured in centimeters.

---

## Overview

This system uses computer vision to:
- Detect the contour of an A4 sheet in an image or video frame
- Warp the image to a top-down view using perspective transformation
- Detect any quadrilateral objects placed on the A4 sheet
- Calculate and display their real-world dimensions (width and height in cm)

---

## Project Structure

```
.
├── main.py         # Main application logic
├── utils.py        # Helper functions: contour detection, warping, measurement
├── test_img.jpg    # Sample input image (can switch to webcam)
```

---

## Requirements

Install the required libraries with:

```bash
pip install opencv-python numpy
```

---

## Usage

### Using an Image (default):
1. Place the object on an A4 sheet and take a picture.
2. Set the image path in `main.py`:
   ```python
   path = 'test_img.jpg'
   webCam = False
   ```
3. Run the script:
   ```bash
   python main.py
   ```

### Using a Webcam:
1. Set webcam mode:
   ```python
   webCam = True
   ```
2. Run the script:
   ```bash
   python main.py
   ```

---

## How It Works

1. Convert the image to grayscale, blur it, and apply Canny edge detection.
2. Find the largest rectangular contour (A4 sheet).
3. Warp the image to get a top-down "scanned" view of the A4 paper.
4. Detect smaller quadrilateral contours inside the warped view.
5. Measure distances between points and convert to centimeters using A4 as the scale.
6. Display results with contours, arrows, and labels on the image.

---

## Customization

- **Scale**: Adjust to fine-tune A4 paper dimensions (default: 210x297 mm)
  ```python
  scale = 2  # Results in wPap = 420, hPap = 594
  ```
- **Contour sensitivity**: Tweak `minArea` and `cThr` in `getContours()`.

---

## Output

- Original image and warped top-down A4 view are displayed
- Detected objects are outlined with arrows
- Real-world dimensions (in cm) are annotated on the image

---
