import fastbook
fastbook.setup_book()


from fastai.vision.all import *
path = 'src/resources/images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
#learn.save('cat_classifier')

#learn.load('cat_classifier')

#img = PILImage.create('src/resources/images/car.jpg')
#pred, pred_idx, probs = learn.predict(img)
#print(f"Prediction: {pred}, Probability: {probs[pred_idx]:.4f}")

