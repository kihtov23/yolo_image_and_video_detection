from imageai.Detection import VideoObjectDetection

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('.\\yolo.h5')
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    input_file_path='.\\input_data\cars.mp4',
    output_file_path='.\\output_data\cars_detected',
    log_progress=True
)