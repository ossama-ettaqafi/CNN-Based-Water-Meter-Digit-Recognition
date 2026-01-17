import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

# Charger le modèle
model = load_model("model.keras",compile=False)
print("✅ Modèle chargé")

# Charger l'image du chiffre
img_path = "0.0_fjpovq.jpg"   # image 28x28 idéalement
img = image.load_img(
    img_path,
    target_size=(32, 20),
    color_mode='rgb'
)

# Convertir l'image en array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # devient (1, 32, 20, 3)

# Prédiction
y_pred_probs = model.predict(img_array)  # shape (1, n_classes)
predicted_digit = np.argmax(y_pred_probs, axis=1)[0]
predicted_prob = y_pred_probs[0][predicted_digit]  # probabilité associée

print(f"🧠 Chiffre prédit : {predicted_digit}")
print(f"📊 Probabilité : {predicted_prob*100:.2f}%")

# Affichage de l'image avec probabilité
plt.imshow(img)
plt.title(f"Prediction: {predicted_digit} ({predicted_prob*100:.2f}%)")
plt.axis("off")
plt.show()
