import base64

from src.models import VisionModel


vision = VisionModel()


with open(
    "data/test.jpg",
    "rb"
) as file:

    image_base64 = base64.b64encode(
        file.read()
    ).decode("utf-8")


result = vision.describe(
    image_base64,
    "image"
)

print("\nVISION RESULT")
print("=" * 80)
print(result)