import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from Adama_model import load_and_preprocess_image


class AdamaTester:
    def __init__(self, csv_path, image_dir, model_path="adama_densenet_model.h5"):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()
        # df["filename"] = df["filename"].apply(os.path.basename)
        # df["filename"] = df.apply(lambda row: os.path.join(self.image_dir, get_subfolder(row), row["filename"]), axis=1)
        df["filename"] = df["filename"].apply(lambda x: os.path.join(self.image_dir, x))

        paths = df["filename"].values
        labels = np.array(df[["Psoriasis", "Eczema"]].values.tolist(), dtype=np.float32)
        return paths, labels

    def test(self):
        paths, labels = self.load_data()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load_and_preprocess_image).batch(32)

        loss, acc = self.model.evaluate(ds)
        print(f"\n Final Test Accuracy: {acc:.2%}")

        y_true, y_pred, per_image_accuracies = [], [], []

        # for i, (images, labels_batch) in enumerate(ds):
        #     preds = self.model.predict(images)
        #     for j in range(len(images)):
        #         # # true = labels_batch[j].numpy()
        #         # # pred = np.zeros_like(preds[j])
        #         # # pred[np.argmax(preds[j])] = 1

        #         # # y_true.append(true)
        #         # # y_pred.append(pred)
        #         # # acc_img = accuracy_score(true, pred)
        #         # # per_image_accuracies.append(acc_img)

        #         # # print(f"\nImage {i*32 + j + 1}:")
        #         # # print(f"  True Labels   : {true.astype(int)}")
        #         # # print(f"  Prediction   : {pred}")
        #         # # print(f"  Accuracy     : {acc_img:.2f}")
        #         # top_class = np.argmax(preds[j])
        #         # confidence = preds[j][top_class]

        #         # if confidence < THRESHOLD:
        #         #     predicted_label = "Unknown"
        #         # else:
        #         #     predicted_label = ["Psoriasis", "Eczema"][top_class]

        #         # true_label = ["Psoriasis", "Eczema"][np.argmax(label[j])]
        #         # print(f"Image {i*batch_size+j+1}: Predicted → {predicted_label} | True → {true_label} | Confidence → {confidence:.2f}")

        THRESHOLD = 0.5  # Confidence threshold for declaring 'Unknown'
        CLASS_NAMES = ["Psoriasis", "Eczema"]  # Update if more classes are added

        for i, (images, labels) in enumerate(ds):
            predictions = self.model.predict(images)

            for j in range(len(predictions)):
                probs = predictions[j]  # Softmax output for one image
                top_class_idx = np.argmax(probs)
                confidence = probs[top_class_idx]

                # Determine label
                predicted_label = CLASS_NAMES[top_class_idx] if confidence >= THRESHOLD else "Unknown"
                true_label = CLASS_NAMES[np.argmax(labels[j])]

                # Display results
                prob_str = ", ".join([f"{CLASS_NAMES[k]}: {probs[k]:.2f}" for k in range(len(CLASS_NAMES))])
                print(f"\n Image {i*len(predictions)+j+1}")
                print(f" True Label: {true_label}")
                print(f" Predicted Probabilities → {prob_str}")
                print(f" Final Prediction: {predicted_label} (Confidence: {confidence:.2f})")


        print(f"\n Average Per-Image Accuracy: {np.mean(per_image_accuracies):.2f}")
        print("\n Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Psoriasis", "Eczema"]))
