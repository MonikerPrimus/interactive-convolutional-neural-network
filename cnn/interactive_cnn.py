import torch
import torchvision.transforms as transforms
import PIL
from PIL import ImageDraw
from tkinter import *
from models import MergedCNN

# Load the trained model
model = MergedCNN()
state_dict = torch.load('merged_cnn_model.pth', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Function to preprocess the drawn image
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

# Establish Drawable GUI Class
class ImageDrawer:
    def __init__(self, master, width, height):
        self.master = master
        self.master.title("Draw Digit for Prediction")
        self.width = width
        self.height = height

        self.canvas = Canvas(self.master, width=self.width, height=self.height, bg="black")
        self.canvas.pack(expand=YES, fill=BOTH)

        self.image = PIL.Image.new("L", (self.width, self.height), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.prev_x, self.prev_y = None, None

        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = Button(self.master, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=LEFT)

        self.clear_button = Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=LEFT)

    def paint(self, event):
        x, y = event.x, event.y
        pen_width = 20
        circle_radius = 20  # Choose an appropriate radius for the circle

        if self.prev_x is not None and self.prev_y is not None:
            self.canvas.create_oval(
                x - circle_radius,
                y - circle_radius,
                x + circle_radius,
                y + circle_radius,
                outline="white",
                fill="white",
                width=pen_width,
            )
            self.draw.ellipse(
                [x - circle_radius * 1.2, y - circle_radius * 1.2, x + circle_radius * 1.2, y + circle_radius * 1.2],
                outline="white",
                fill="white",
                width=pen_width * 1.2,
            )

        self.prev_x, self.prev_y = x, y

    def predict_digit(self):
        # Save the drawn canvas as an image
        self.image.save("drawn_digit.png")

        # Load the drawn image using PIL
        drawn_image = PIL.Image.open("drawn_digit.png").convert("L")  # Ensure it's grayscale

        # Preprocess and predict
        with torch.no_grad():
            model_output = model(preprocess_image(drawn_image))

        # Get the predicted label
        predicted_label = torch.argmax(model_output).item()

        print(f"The predicted label is: {predicted_label}")

        # Reset prev_x and prev_y to None
        self.prev_x, self.prev_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("L", (self.width, self.height), "black")
        self.draw = ImageDraw.Draw(self.image)
        # Reset prev_x and prev_y to None
        self.prev_x, self.prev_y = None, None


if __name__ == "__main__":
    root = Tk()
    width, height = 480, 480
    app = ImageDrawer(root, width, height)
    root.mainloop()
