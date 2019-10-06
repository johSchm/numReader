
import numberClassifier as nc
import freeDraw as draw
import Window as win


def predict_number(_classifier, _drawapp):
    """ Predicts the drawn number and outputs the result in the console.
    :param _classifier:
    :param _drawapp:
    """
    if _classifier is not None and _drawapp is not None:
        img = _drawapp.pixel_array()
        prediction = _classifier.predict(img)
        print("Prediction: {}".format(prediction))


if __name__ == '__main__':
    """ Main method.
    """
    classifier = nc.NumberClassifier(load_existing_model=True)
    app = draw.FreeDrawingApp(parent=win.Window(), pensize=30)
    app.run(
        interrupt_method=predict_number, interrupt_interval=5,
        _classifier=classifier, _drawapp=app)
