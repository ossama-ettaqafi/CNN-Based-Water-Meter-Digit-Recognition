from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import re
from collections import Counter

# ================== CONFIGURATION TESSERACT ==================
# Chemin vers l'exécutable Tesseract (à adapter selon votre installation)
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# ================== APPLICATION FLASK ==================
app = Flask(__name__)

# Dossiers pour stocker les fichiers uploadés et les images de débogage
UPLOAD_FOLDER = "uploads"
DEBUG_FOLDER = "debug"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DEBUG_FOLDER"] = DEBUG_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Taille maximale de fichier : 16MB

# =====================================================
# DÉTECTION PRÉCISE DE TOUS LES CHIFFRES DANS LA ROI
# =====================================================
def detect_precise_digits(image_roi, roi_coords, full_image):
    """Détecte précisément TOUS les chiffres individuels dans la ROI"""
    x1, y1, x2, y2 = roi_coords
    h_roi, w_roi = image_roi.shape
    
    # Sauvegarde de la ROI originale pour débogage
    debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_original_roi.jpg")
    cv2.imwrite(debug_path, image_roi)
    
    # Étape 1 : Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_roi)
    
    # Étape 2 : Seuillage adaptatif
    binary = cv2.adaptiveThreshold(enhanced, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Sauvegarde de l'image binaire pour débogage
    binary_debug_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_binary.jpg")
    cv2.imwrite(binary_debug_path, binary)
    
    # Étape 3 : Nettoyage de l'image binaire
    kernel_clean = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    
    # Étape 4 : Recherche des contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_info = []
    
    # Paramètres ORIGINAUX (garder les mêmes seuils)
    min_digit_height = h_roi * 0.4
    max_digit_height = h_roi * 0.9
    min_digit_width = w_roi * 0.05
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filtrage ORIGINAL
        if (min_digit_height < h < max_digit_height and
            w > min_digit_width and
            area > 100 and
            0.3 < w/h < 1.0):
            
            # Calcul de la compacité
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                
                # Plage ORIGINALE de compacité
                if 0.2 < compactness < 0.8:
                    
                    # Padding
                    pad = 2
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(w_roi, x + w + pad)
                    y_end = min(h_roi, y + h + pad)
                    
                    # Extraction de la région
                    digit_roi = enhanced[y_start:y_end, x_start:x_end]
                    
                    # Redimensionnement
                    if digit_roi.size > 0:
                        target_height = 60
                        aspect_ratio = w / h
                        target_width = int(target_height * aspect_ratio)
                        digit_roi_resized = cv2.resize(digit_roi, 
                                                      (max(20, target_width), target_height),
                                                      interpolation=cv2.INTER_CUBIC)
                        
                        # Application d'un léger flou gaussien
                        digit_roi_resized = cv2.GaussianBlur(digit_roi_resized, (1, 1), 0)
                        
                        # Binarisation spécifique pour l'OCR
                        _, digit_binary = cv2.threshold(digit_roi_resized, 0, 255, 
                                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # Essayer plusieurs configurations OCR
                        digit_text = ""
                        configs = [
                            "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789",
                            "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789",
                            "--psm 10 --oem 3"
                        ]
                        
                        for config in configs:
                            digit_text = pytesseract.image_to_string(digit_binary, 
                                                                    config=config).strip()
                            if digit_text and digit_text[0].isdigit():
                                break
                        
                        # Si l'OCR échoue, essayer avec l'image non binaire
                        if not digit_text or not digit_text[0].isdigit():
                            for config in configs:
                                digit_text = pytesseract.image_to_string(digit_roi_resized, 
                                                                        config=config).strip()
                                if digit_text and digit_text[0].isdigit():
                                    break
                        
                        if digit_text and digit_text[0].isdigit():
                            digit = digit_text[0]
                            
                            # Calcul de la confiance
                            aspect_score = 1.0 - abs(0.6 - w/h)
                            height_score = 1.0 - abs(0.6 - h/h_roi)
                            confidence = min(95.0, 70.0 + (aspect_score + height_score) * 12.5)
                            
                            # Coordonnées absolues
                            abs_x = x1 + x_start
                            abs_y = y1 + y_start
                            abs_w = x_end - x_start
                            abs_h = y_end - y_start
                            
                            digit_info.append({
                                'x': abs_x,
                                'y': abs_y,
                                'w': abs_w,
                                'h': abs_h,
                                'center_x': abs_x + abs_w // 2,
                                'digit': digit,
                                'confidence': confidence,
                                'contour_area': area,
                                'aspect_ratio': w/h
                            })
                            
                            # Sauvegarde de l'image de débogage
                            digit_debug_path = os.path.join(DEBUG_FOLDER, 
                                                          f"{debug_timestamp}_digit_{len(digit_info)}.jpg")
                            cv2.imwrite(digit_debug_path, digit_roi_resized)
    
    # Tri des chiffres par position horizontale
    digit_info.sort(key=lambda d: d['center_x'])
    
    # Filtrage des doublons - AVEC PLUS DE TOLÉRANCE
    filtered_digits = []
    if digit_info:
        filtered_digits.append(digit_info[0])
        for i in range(1, len(digit_info)):
            current = digit_info[i]
            previous = filtered_digits[-1]
            
            # Vérifier si les chiffres sont trop proches
            overlap_threshold = previous['w'] * 0.3
            distance = current['center_x'] - previous['center_x']
            
            if distance > overlap_threshold:
                filtered_digits.append(current)
            elif current['confidence'] > previous['confidence'] + 5:
                # Si chevauchement significatif, garder celui avec plus de confiance
                filtered_digits[-1] = current
            # Sinon, ignorer le doublon (déjà gardé le premier)
    
    print(f"Détection de {len(filtered_digits)} chiffre(s) dans la ROI")
    return filtered_digits, debug_timestamp

def draw_precise_boxes(full_image, digit_info, reading, roi_coords=None):
    """Dessine des boîtes vertes précises autour des chiffres détectés"""
    output = full_image.copy()
    
    # Dessiner la ROI si fournie
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 100, 255), 2)  # Orange pour la ROI
        cv2.putText(output, "Zone de recherche des chiffres", 
                   (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 100, 255), 2)
    
    # Dessiner des boîtes vertes précises pour chaque chiffre détecté
    for i, digit_data in enumerate(digit_info):
        x, y, w, h = digit_data['x'], digit_data['y'], digit_data['w'], digit_data['h']
        
        # Dessiner un rectangle vert
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Dessiner un arrière-plan rempli pour l'étiquette
        label = f"{digit_data['digit']}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(output, 
                     (x, y - text_size[1] - 10),
                     (x + text_size[0] + 10, y),
                     (0, 255, 0), -1)
        
        # Étiquette avec la valeur du chiffre en noir
        cv2.putText(output, label,
                   (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 0), 2)
        
        # Ajouter la confiance en dessous de la boîte
        conf_text = f"{digit_data['confidence']:.0f}%"
        cv2.putText(output, conf_text,
                   (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 200, 0), 1)
    
    # Dessiner la lecture complète en haut
    if reading:
        # Arrière-plan pour la lecture
        text_size = cv2.getTextSize(f"Lecture: {reading}", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.rectangle(output, 
                     (20, 20),
                     (20 + text_size[0] + 20, 20 + text_size[1] + 20),
                     (0, 0, 0), -1)
        
        cv2.putText(output, f"Lecture: {reading}",
                   (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 255), 3)
    
    return output

# =====================================================
# EXTRACTION PRÉCISE DE TOUS LES CHIFFRES
# =====================================================
def extract_precise_meter_reading(image):
    """Extrait TOUS les chiffres avec détection précise (ROI ORIGINALE)"""
    h, w = image.shape[:2]
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ROI ORIGINALE (exactement comme dans le code original)
    roi_y1, roi_y2 = int(h * 0.28), int(h * 0.38)  # Bande verticale très étroite
    roi_x1, roi_x2 = int(w * 0.35), int(w * 0.65)  # Centrée horizontalement
    
    # Extraction de la ROI
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
    
    # Détection précise de TOUS les chiffres dans cette ROI étroite
    digit_info, debug_timestamp = detect_precise_digits(roi, roi_coords, image)
    
    # Si nous détectons des chiffres dans cette ROI étroite
    if digit_info:
        # Trier par position X
        digit_info.sort(key=lambda d: d['center_x'])
        
        # Construction de la lecture à partir des chiffres détectés
        detected_digits = ''.join([d['digit'] for d in digit_info if d['digit'].isdigit()])
        
        # Si nous avons au moins 5 chiffres (basé sur votre sortie "Chiffres détectés: 5")
        if len(detected_digits) >= 5:
            # Prendre tous les chiffres détectés
            reading = detected_digits
            
            # Calculer la confiance moyenne
            avg_confidence = sum(d['confidence'] for d in digit_info) / len(digit_info)
            confidence = min(95.0, avg_confidence * 0.95)
            
            print(f"Détection dans ROI étroite: {len(digit_info)} chiffres -> {reading}")
        else:
            # Fallback: OCR sur toute la ROI
            config_full = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
            digits_text = pytesseract.image_to_string(roi, config=config_full).replace(" ", "")
            
            if len(digits_text) >= 5:
                reading = digits_text
                confidence = 70.0
                print(f"OCR sur ROI: {reading}")
            else:
                reading = ""
                confidence = 30.0
                print(f"Pas assez de chiffres détectés")
    else:
        # Fallback complet
        config_full = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
        digits_text = pytesseract.image_to_string(roi, config=config_full).replace(" ", "")
        
        if len(digits_text) >= 5:
            reading = digits_text
            confidence = 70.0
            print(f"OCR uniquement: {reading}")
        else:
            # Essayer une configuration OCR alternative
            config_alt = "--psm 8 --oem 3"
            alt_text = pytesseract.image_to_string(roi, config=config_alt)
            
            # Rechercher un motif à plusieurs chiffres
            match = re.search(r'(\d{5,})', alt_text)
            if match:
                reading = match.group(1)
                confidence = 65.0
                print(f"Motif trouvé: {reading}")
            else:
                # Rechercher toute séquence de chiffres
                all_digits = re.findall(r'\d', alt_text)
                if len(all_digits) >= 5:
                    reading = ''.join(all_digits)
                    confidence = 60.0
                    print(f"Extraction depuis le texte: {reading}")
                else:
                    reading = ""
                    confidence = 30.0
                    print(f"Aucune lecture fiable trouvée")
    
    # Création de la visualisation avec les boîtes précises
    output_image = draw_precise_boxes(image, digit_info, reading, roi_coords)
    
    # Ajout du texte de confiance et d'information
    cv2.putText(output_image, f"Confiance: {confidence:.1f}%",
               (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 0), 2)
    
    cv2.putText(output_image, f"Chiffres détectés: {len(digit_info)}",
               (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 200, 255), 2)
    
    # Afficher tous les chiffres détectés
    if digit_info:
        all_digits_str = ''.join([d['digit'] for d in digit_info])
        cv2.putText(output_image, f"Tous chiffres: {all_digits_str}",
                   (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 200, 0), 2)
    
    # Ajout du timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output_image, f"Traité le: {timestamp}",
               (20, h-20), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)
    
    return reading, confidence, output_image, digit_info, debug_timestamp

# =====================================================
# TRAITEMENT PRINCIPAL
# =====================================================
def process_image(image_path):
    """Fonction de traitement principale (ROI ORIGINALE)"""
    # Lecture de l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Impossible de lire l'image")
    
    h, w = image.shape[:2]
    
    # Extraction de la lecture avec ROI ORIGINALE
    reading, confidence, output_image, digit_info, debug_timestamp = extract_precise_meter_reading(image)
    
    print(f"Lecture finale: {reading}, Confiance: {confidence:.1f}%")
    
    # Sauvegarde de l'image résultat
    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_path, output_image)
    
    # Préparation des informations détaillées sur les chiffres pour la réponse
    digit_details = []
    reading_digits = list(reading) if reading else []
    
    for i, expected_digit in enumerate(reading_digits):
        detected = False
        digit_confidence = 0.0
        box_info = None
        
        # Essayer de trouver un chiffre détecté à cette position
        if i < len(digit_info):
            detected_digit = digit_info[i]
            detected = (detected_digit['digit'] == expected_digit)
            digit_confidence = detected_digit['confidence']
            box_info = {
                'x': detected_digit['x'],
                'y': detected_digit['y'],
                'width': detected_digit['w'],
                'height': detected_digit['h']
            }
        else:
            # Estimation de la confiance basée sur la position et la confiance globale
            digit_confidence = confidence * (0.9 - (i * 0.05))
        
        status = '✓' if detected else '⚠' if digit_confidence > 60 else '✗'
        
        digit_details.append({
            'position': i + 1,
            'digit': expected_digit,
            'confidence': f"{digit_confidence:.1f}%",
            'detected': detected,
            'status': status,
            'box_coordinates': box_info
        })
    
    # Information sur tous les chiffres détectés
    all_digits_str = ''.join([d['digit'] for d in digit_info]) if digit_info else ""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'reading': reading,
        'confidence': f"{confidence:.1f}%",
        'image_url': f"/uploads/{filename}",
        'debug_url': f"/debug/{debug_timestamp}_original_roi.jpg",
        'digit_details': digit_details,
        'digit_count': len(reading_digits),
        'total_digits_detected': len(digit_info),
        'all_detected_digits': all_digits_str,  # Tous les chiffres détectés dans la ROI
        'timestamp': timestamp,
        'debug_timestamp': debug_timestamp
    }

# =====================================================
# ROUTES FLASK (inchangées)
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/debug/<filename>")
def debug_file(filename):
    return send_from_directory(DEBUG_FOLDER, filename)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "Aucune image fournie"}), 400
        
        # Validation du type de fichier
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return jsonify({"error": "Type de fichier invalide. Utilisez PNG, JPG ou BMP"}), 400
        
        # Sauvegarde du fichier uploadé
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        input_name = f"input_{timestamp}.jpg"
        input_path = os.path.join(UPLOAD_FOLDER, input_name)
        file.save(input_path)
        
        # Traitement de l'image
        result_data = process_image(input_path)
        
        return jsonify({
            "success": True,
            **result_data
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Erreur: {e}\n{error_trace}")
        return jsonify({"error": f"Erreur de traitement: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route("/debug_images")
def list_debug_images():
    debug_files = []
    for file in os.listdir(DEBUG_FOLDER):
        if file.endswith('.jpg'):
            debug_files.append({
                'name': file,
                'url': f'/debug/{file}',
                'size': os.path.getsize(os.path.join(DEBUG_FOLDER, file))
            })
    
    return jsonify({
        'debug_files': sorted(debug_files, key=lambda x: x['name'], reverse=True)[:10]
    })

# =====================================================
if __name__ == "__main__":
    print("🚰 OCR PRÉCIS POUR COMPTEUR D'EAU - ROI ORIGINALE")
    print("🌐 http://127.0.0.1:5000")
    print("📁 Dossier d'upload:", os.path.abspath(UPLOAD_FOLDER))
    print("🎯 Configuration:")
    print("   - ROI étroite et ciblée (28%-38% hauteur, 35%-65% largeur)")
    print("   - Détection de TOUS les chiffres dans cette ROI spécifique")
    print("   - Paramètres de détection originaux conservés")
    print("   - Sortie inclut 'all_detected_digits' avec tous les chiffres trouvés")
    
    app.run(debug=True, host='0.0.0.0', port=5000)