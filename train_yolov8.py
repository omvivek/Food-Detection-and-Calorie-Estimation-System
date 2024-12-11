from ultralytics import YOLO 

def train_yolov8():
    print("Starting YOLOv8 training...")
    # Load YOLOv8 model (pretrained weights)
    model = YOLO("yolov8s.pt")  # yolov8n, yolov8s, yolov8m, etc. based on your needs

    # Train the model
    model.train(
        data="E:/Food Detection & Calorie Estimation System/dataset/data.yaml",  # Absolute path to data.yaml
        epochs=50,           # Number of training epochs
        imgsz=640,           # Image size
        batch=16,            # Batch size (adjust based on your GPU)
        name="custom_yolov8" # Name of the training run
    )
    print("Training completed!")

def evaluate_yolov8():
    print("Evaluating the model...")
    model = YOLO("runs/detect/custom_yolov8/weights/best.pt")  # Path to the best model
    results = model.val(data="E:/Food Detection & Calorie Estimation System/dataset/data.yaml")  # Evaluate using validation data
    print("Evaluation results:", results)

def predict_yolov8():
    print("Running inference on test images...")
    model = YOLO("runs/detect/custom_yolov8/weights/best.pt")  # Path to the trained model
    results = model.predict(source="path/to/test/images", save=True)  # Path to your test images
    print("Inference completed. Results saved.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], required=True,
                        help="Mode to run: train, evaluate, predict")
    args = parser.parse_args()

    if args.mode == 'train':
        train_yolov8()
    elif args.mode == 'evaluate':
        evaluate_yolov8()
    elif args.mode == 'predict':
        predict_yolov8()
