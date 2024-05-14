from model.od_model import model
from utils.data_loader_bbox import train_images, train_annotations

# Train the model
num_epochs = 3  # Adjust the number of epochs as needed
batch_size = 1  # Adjust the batch size as needed

model.fit(train_images, train_annotations, epochs=num_epochs, batch_size=batch_size)

# Save the trained model
model.save('object_detection_model.h5')