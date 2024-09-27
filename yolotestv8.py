from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
#results = model(["bus.jpg", "DunnesStoresImmuneClosedCups450gLabel.jpg"])  # return a list of Results objects

imgfile = "DunnesStoresImmuneClosedCups450gLabel.jpg"

# results = model("bus.jpg")  # return a list of Results objects
results = model(imgfile)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    #result.save(filename="result.jpg")  # save to disk
    #print(result.obb)  # print results to screen

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")