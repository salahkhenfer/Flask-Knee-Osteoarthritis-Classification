from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

import numpy as np

app = Flask(__name__)

dic = {
    0: " تشير إلى حالة أو مستوى عادي أو طبيعي دون وجود مشكلة كبيرة.",
    1: " تشير إلى حالة تثير شكوكًا أو عدم اليقين بشأنها، وغالبًا ما تكون هذه الحالة تحتاج إلى مزيد من التقييم أو الاهتمام.",
    2: " تشير إلى حالة أو مستوى بسيط أو خفيف من مشكلة معينة، وعادة ما يكون الأثر الضار محدودًا",
    3: " تشير إلى حالة أو مستوى متوسط أو معتدل من مشكلة معينة، وقد يكون له تأثير أكبر من الحالة الخفيفة ولكن ليس بالشكل الكبير.",
    4: " تشير إلى حالة أو مستوى شديد أو كبير من مشكلة معينة، وغالبًا ما يكون له تأثير كبير ويتطلب اهتمامًا وعلاجًا فوريً",
}

# Image Size
img_size = 256
model = load_model("model.h5")

model.make_predict_function()


def predict_label(img_path):
    img = Image.open(img_path).convert("L")  # Open image in grayscale mode
    img = img.resize((img_size, img_size))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, img_size, img_size, 1)
    probabilities = model.predict(img_array)[0]
    predicted_class = dic[int(np.argmax(probabilities))]
    return predicted_class


# routes
@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        img = request.files["file"]
        img_path = "uploads/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        print(p)
        return str(p).lower()


if __name__ == "__main__":
    app.run(debug=True)
