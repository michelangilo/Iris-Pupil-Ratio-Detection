<h1>Project</h1>
This project calculates the Iris to Pupil ratio in:
<ol>
  <li>Real Time Live Camera using python mediapipe library <b>(Detection Using Live Camera)</b></li>
  <li>the dataset using python mediapipe library <b>(Detection Using mediapipe)</b></li>
  <li>the dataset using YOLO-MobileNetV2 model <b>(Detection Using YOLO-MobileNetV2)</b></li>
</ol>

<h2>Dataset</h2>
The dataset given consists of 450 images.
<br>
Irispupil is annotated in Tensorflow Object Detection format.
<br>
The following pre-processing was applied to each image:
<ol>
  <li>Auto-orientation of pixel data (with EXIF-orientation stripping)</li>
  <li>Resize to 640x640 (Stretch)</li>
</ol>
<ul>
  <li>No image augmentation techniques were applied.</li>
  <li>The dataset is divided into three parts: 310 training images, 90 validation images and 45 test images of the human eye.</li>
  <li>Each part has its own <b>_annotations.csv</b> stored in the respective folder. </li>
  <li><b>Detection Using YOLO-MobileNetV2</b> contains all three folders as the model is trained, validated and tested on the dataset.</li>
  <li><b>Detection Using mediapipe</b> contains the test folder, upon running <b>model.py</b>, it creates an <b>images</b> folder with the resultant test images.</li>
</ul>
