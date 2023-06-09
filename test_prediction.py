import coremltools as ct
from PIL import Image

# Load the model
model = ct.models.MLModel('densenetmodel.mlmodel')

example_image = Image.open("happy.png").resize((48, 48))
predictions = model.predict(example_image)

print(predictions)
