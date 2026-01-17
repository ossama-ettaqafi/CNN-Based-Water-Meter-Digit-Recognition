import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# 1. Chargement du modèle
model = load_model("model.keras", compile=False)

def predict_digit(img_digit):
    """Prédiction d'un chiffre avec prétraitement correct pour modèle RGB"""
    # S'assurer que l'image est en RGB (3 canaux)
    if len(img_digit.shape) == 2:  # Si grayscale
        img_digit = cv2.cvtColor(img_digit, cv2.COLOR_GRAY2RGB)
    
    # Redimensionner à la taille attendue par le modèle (20x32 pour RGB)
    img_digit = cv2.resize(img_digit, (20, 32))
    
    # Normaliser les valeurs des pixels
    img_digit = img_digit.astype('float32') / 255.0
    
    # Ajouter la dimension du batch
    img_digit = np.expand_dims(img_digit, axis=0)
    
    # Prédiction
    probs = model.predict(img_digit, verbose=0)
    return np.argmax(probs), np.max(probs)

# 2. Chargement de l'image
image_path = "image.png"
image = cv2.imread(image_path)
if image is None:
    print("Erreur: Impossible de charger l'image")
    exit()

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Amélioration du contraste
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# 4. Définition de la ROI (vous pouvez ajuster ces valeurs selon votre image)
h_img, w_img = gray.shape
roi_y, roi_h = int(h_img*0.35), int(h_img*0.12)  # Ajuster selon votre image
roi_x, roi_w = int(w_img*0.48), int(w_img*0.35)   # Ajuster selon votre image
roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
roi_color = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

# 5. Traitement pour extraire les chiffres individuellement
# Seuillage pour isoler les chiffres
_, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Nettoyage avec morphologie
kernel = np.ones((2,2), np.uint8)
roi_clean = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)
roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel)

# 6. Détection des chiffres individuels
# Trouver les contours
contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrer les contours par taille (pour éliminer le bruit)
digit_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Filtrer par taille (ajuster selon votre image)
    if h > roi_h * 0.4 and w > roi_w * 0.05 and w < roi_w * 0.2:
        digit_contours.append((x, y, w, h))

# Trier les contours de gauche à droite
digit_contours.sort(key=lambda x: x[0])

# Vérifier si nous avons détecté des chiffres
if len(digit_contours) == 0:
    print("Aucun chiffre détecté! Ajustez les paramètres de la ROI.")
    exit()

# 7. Prédiction chiffre par chiffre
compteur_final = ""
confidences = []

# Si nous avons plus de 8 chiffres, ne prendre que les 8 premiers
if len(digit_contours) > 8:
    digit_contours = digit_contours[:8]

for i, (x, y, w, h) in enumerate(digit_contours):
    # Extraire le chiffre individuel de l'image ROI (originale, pas seuillée)
    digit_img = roi[y:y+h, x:x+w]
    
    # Pour la visualisation, on peut aussi extraire la version couleur
    digit_color = roi_color[y:y+h, x:x+w]
    
    # Améliorer le contraste du chiffre
    _, digit_processed = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Ajouter une marge blanche autour du chiffre
    margin = 5
    digit_with_margin = np.zeros((h + 2*margin, w + 2*margin), dtype=np.uint8)
    digit_with_margin[margin:margin+h, margin:margin+w] = digit_processed
    
    # Inverser si nécessaire (chiffres blancs sur fond noir)
    # La plupart des modèles attendent des chiffres blancs sur fond noir
    if np.mean(digit_with_margin) < 127:  # Si l'image est principalement noire
        digit_with_margin = cv2.bitwise_not(digit_with_margin)
    
    # Convertir en RGB pour le modèle
    digit_rgb = cv2.cvtColor(digit_with_margin, cv2.COLOR_GRAY2RGB)
    
    # Prédiction
    digit, confidence = predict_digit(digit_rgb)
    compteur_final += str(digit)
    confidences.append(confidence)
    
    # Dessiner pour visualisation
    cv2.rectangle(output, 
                 (roi_x + x, roi_y + y), 
                 (roi_x + x + w, roi_y + y + h), 
                 (0, 255, 0), 2)
    cv2.putText(output, str(digit), 
               (roi_x + x, roi_y + y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 8. Affichage des résultats
print(f"Nombre de chiffres détectés: {len(digit_contours)}")
print(f"Valeur lue : {compteur_final}")
if confidences:
    print(f"Confiance moyenne : {np.mean(confidences):.2%}")

# 9. Visualisation détaillée
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

# ROI et traitement global
axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Image originale")
axes[0, 0].axis('off')

axes[0, 1].imshow(roi, cmap='gray')
axes[0, 1].set_title("ROI (niveaux de gris)")
axes[0, 1].axis('off')

axes[0, 2].imshow(roi_thresh, cmap='gray')
axes[0, 2].set_title("ROI seuillée")
axes[0, 2].axis('off')

axes[0, 3].imshow(roi_clean, cmap='gray')
axes[0, 3].set_title("ROI nettoyée")
axes[0, 3].axis('off')

# Résultat final
axes[1, 0].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f"Résultat: {compteur_final}")
axes[1, 0].axis('off')

# Afficher chaque chiffre extrait
for i in range(min(6, len(digit_contours))):
    x, y, w, h = digit_contours[i]
    digit_img = roi[y:y+h, x:x+w]
    _, digit_processed = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Ajouter une marge et convertir en RGB pour l'affichage
    margin = 5
    digit_with_margin = np.zeros((h + 2*margin, w + 2*margin), dtype=np.uint8)
    digit_with_margin[margin:margin+h, margin:margin+w] = digit_processed
    
    if np.mean(digit_with_margin) < 127:
        digit_with_margin = cv2.bitwise_not(digit_with_margin)
    
    digit_rgb = cv2.cvtColor(digit_with_margin, cv2.COLOR_GRAY2RGB)
    
    # Redimensionner pour l'affichage
    digit_display = cv2.resize(digit_rgb, (40, 64))
    
    row = 1 + (i // 3)  # 1 pour la première rangée, 2 pour la suivante
    col = 1 + (i % 3)   # 1, 2, ou 3
    
    if row < 3:  # S'assurer qu'on ne dépasse pas la grille
        axes[row, col].imshow(digit_display)
        axes[row, col].set_title(f"Chiffre {i}: {compteur_final[i]}")
        axes[row, col].axis('off')

# Masquer les axes non utilisés
for i in range(3):
    for j in range(4):
        if not axes[i, j].has_data():
            axes[i, j].axis('off')

plt.tight_layout()
plt.show()

# 10. Afficher aussi l'image finale seule
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Compteur détecté: {compteur_final}")
plt.axis('off')
plt.show()