from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Load a model
    model = YOLO(r"D:\projects\trials\data science\computer vision- YOLO - series\custom dataset weights\best_model_trained.pt")  # load a pretrained model (recommended for training)

    # Use the model
    results = model(source=0,show=True,save=True)
