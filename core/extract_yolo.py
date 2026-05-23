from ultralytics import YOLO
model = YOLO("yolo26n.pt")  # Load model
model.export(format="saved_model", keras = True) # Export to TF SavedModel


# Load the exported SavedModel
