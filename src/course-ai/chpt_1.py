from fastai.vision.all import *
from fastai.text.all import *
import fastbook

def train_cat_classifier():
    fastbook.setup_book()
    path = untar_data(URLs.PETS)/'images'

    def is_cat(x): return x[0].isupper()

    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224))

    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)
    learn.save('cat_classifier')


def test_cat_classifier(imagePath = 'src/resources/images/lebron.jpg'):
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.load('cat_classifier')
    img = PILImage.create(imagePath)
    pred, pred_idx, probs = learn.predict(img)
    print(f"Prediction: {pred}, Probability: {probs[pred_idx]:.4f}")

###############################################################################

def train_something():
    dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(4, 1e-2)
    learn.save()
    learn.show_results(max_n=6, figsize=(7,8))

train_something()