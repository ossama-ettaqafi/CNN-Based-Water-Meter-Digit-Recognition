from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import re
from collections import Counter
import logging

# ================== CONFIGURATION TESSERACT ==================
# Chemin vers l'exécutable Tesseract (à adapter selon votre installation)
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# ================== CONFIGURATION LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_log.txt'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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
# FONCTIONS AUXILIAIRES
# =====================================================
def find_digit_region(image):
    """Trouve automatiquement la région contenant les chiffres"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sauvegarde pour débogage
    debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_find_region_gray.jpg")
    cv2.imwrite(debug_path, gray)
    
    # Appliquer un seuil adaptatif
    thresh = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Sauvegarde du seuil pour débogage
    thresh_path = os.path.join(DEBUG_FOLDER, f"{debug_timestamp}_find_region_thresh.jpg")
    cv2.imwrite(thresh_path, thresh)
    
    # Nettoyage morphologique
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours par taille et forme
    digit_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # ================== CONDITION MODIFIÉE ==================
        # Critères pour des chiffres (éviter les chiffres trop grands)
        max_height_ratio = 0.25  # Maximum 25% de la hauteur de l'image
        min_aspect_ratio = 0.3
        max_aspect_ratio = 1.2
        
        if (h < image.shape[0] * max_height_ratio and  # Éviter les très grands chiffres
            h > image.shape[0] * 0.05 and  # Hauteur minimale
            w > image.shape[1] * 0.02 and   # Largeur minimale
            area > 100 and                  # Aire minimale
            min_aspect_ratio < aspect_ratio < max_aspect_ratio):  # Ratio raisonnable
            digit_contours.append((x, y, w, h))
        # =====================================================
    
    logger.info(f"Nombre de contours détectés comme chiffres: {len(digit_contours)}")
    
    # Si on a trouvé des contours, calculer la région englobante
    if digit_contours:
        x_coords = [x for x, _, _, _ in digit_contours]
        y_coords = [y for _, y, _, _ in digit_contours]
        w_coords = [w for _, _, w, _ in digit_contours]
        h_coords = [h for _, _, _, h in digit_contours]
        
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max([x + w for x, w in zip(x_coords, w_coords)])
        y2 = max([y + h for y, h in zip(y_coords, h_coords)])
        
        # Ajouter une marge
        margin_x = int(image.shape[1] * 0.1)
        margin_y = int(image.shape[0] * 0.1)
        
        region = (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(image.shape[1], x2 + margin_x),
            min(image.shape[0], y2 + margin_y)
        )
        
        logger.info(f"Région détectée automatiquement: {region}")
        return region
    
    logger.info("Aucune région détectée automatiquement, utilisation de la ROI par défaut")
    return None

def validate_reading(reading, expected_length=None):
    """Valide et corrige si nécessaire la lecture"""
    if not reading:
        return reading
    
    # S'assurer que ce sont bien des chiffres
    digits = re.findall(r'\d', reading)
    
    logger.info(f"Validation: {len(digits)} chiffres trouvés dans '{reading}'")
    
    # Si aucune longueur attendue spécifiée, retourner tous les chiffres
    if expected_length is None:
        result = ''.join(digits)
        logger.info(f"Aucune longueur attendue, retourne tous les chiffres: {result}")
        return result
    
    if len(digits) < expected_length:
        # Si trop peu de chiffres, retourner ce qu'on a
        result = ''.join(digits)
        logger.warning(f"Trop peu de chiffres: {len(digits)} < {expected_length}")
        return result
    elif len(digits) > expected_length:
        # Si trop de chiffres, prendre les premiers
        result = ''.join(digits[:expected_length])
        logger.warning(f"Trop de chiffres: {len(digits)} > {expected_length}, tronqué à {result}")
        return result
    else:
        result = ''.join(digits)
        logger.info(f"Lecture validée: {result}")
        return result

# =====================================================
# DÉTECTION PRÉCISE DE TOUS LES CHIFFRES DANS LA ROI
# =====================================================
def detect_precise_digits(image_roi, roi_coords, full_image):
    """Détecte précisément TOUS les chiffres individuels dans la ROI"""
    x1, y1, x2, y2 = roi_coords
    h_roi, w_roi = image_roi.shape
    
    logger.info(f"Détection dans ROI: {roi_coords}, taille: {w_roi}x{h_roi}")
    
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
    
    logger.info(f"Nombre de contours trouvés: {len(contours)}")
    
    digit_info = []
    
    # ================== PARAMÈTRES AJOUTÉS ==================
    # Hauteur maximale pour un chiffre (ignorer les chiffres trop grands)
    max_digit_height_ratio = 0.6  # 60% de la hauteur de la ROI
    max_digit_height_pixels = h_roi * max_digit_height_ratio
    
    # Hauteur minimale pour un chiffre
    min_digit_height = h_roi * 0.3
    min_digit_width = w_roi * 0.03
    
    # Ratio largeur/hauteur maximal pour éviter les chiffres trop étroits/hauts
    max_aspect_ratio = 0.8  # w/h maximal (éviter les chiffres trop hauts)
    min_aspect_ratio = 0.25  # w/h minimal
    
    logger.info(f"Seuils de détection - Hauteur max: {max_digit_height_pixels:.1f}px, "
                f"Ratio max: {max_aspect_ratio}")
    # =====================================================
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # ================== FILTRAGE AMÉLIORÉ ==================
        # Condition pour ignorer les chiffres trop grands (tall numbers)
        is_too_tall = h > max_digit_height_pixels
        is_too_narrow_tall = aspect_ratio < min_aspect_ratio and h > h_roi * 0.5
        
        # Vérifier si le contour ressemble à un chiffre (pas un symbole ou autre)
        if (not is_too_tall and 
            not is_too_narrow_tall and
            min_digit_height < h < max_digit_height_pixels and
            w > min_digit_width and
            area > 80 and
            min_aspect_ratio < aspect_ratio < max_aspect_ratio):
        # =====================================================
            
            # Calcul de la compacité
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                
                # Filtre de compacité
                if 0.15 < compactness < 0.9:
                    
                    # Padding
                    pad = 3
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(w_roi, x + w + pad)
                    y_end = min(h_roi, y + h + pad)
                    
                    # Extraction de la région
                    digit_roi = enhanced[y_start:y_end, x_start:x_end]
                    
                    if digit_roi.size == 0:
                        continue
                    
                    # Redimensionnement
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
                    
                    # Essayer plusieurs configurations OCR avec plus de caractères
                    digit_text = ""
                    configs = [
                        "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789m³",
                        "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789m³",
                        "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789",
                        "--psm 8 --oem 3"
                    ]
                    
                    for config in configs:
                        try:
                            digit_text = pytesseract.image_to_string(digit_binary, 
                                                                    config=config).strip()
                            if digit_text and (digit_text[0].isdigit() or digit_text[0] in 'm³'):
                                break
                        except Exception as e:
                            logger.warning(f"Erreur OCR avec config {config}: {e}")
                    
                    # Si l'OCR échoue, essayer avec l'image non binaire
                    if not digit_text or not (digit_text[0].isdigit() or digit_text[0] in 'm³'):
                        for config in configs:
                            try:
                                digit_text = pytesseract.image_to_string(digit_roi_resized, 
                                                                        config=config).strip()
                                if digit_text and (digit_text[0].isdigit() or digit_text[0] in 'm³'):
                                    break
                            except Exception as e:
                                logger.warning(f"Erreur OCR avec image non binaire: {e}")
                    
                    if digit_text and (digit_text[0].isdigit() or digit_text[0] in 'm³'):
                        digit = digit_text[0] if digit_text[0] in '0123456789m³' else '?'
                        
                        # Calcul de la confiance avec pénalité pour les hauteurs extrêmes
                        height_ratio = h / h_roi
                        height_score = 1.0 - min(0.5, abs(0.4 - height_ratio))  # Meilleur score pour 40% de la hauteur ROI
                        
                        # Pénalité supplémentaire si trop haut
                        if height_ratio > 0.5:
                            height_score *= 0.7
                        
                        aspect_score = 1.0 - min(0.5, abs(0.6 - w/h))
                        compactness_score = 1.0 - min(0.5, abs(0.4 - compactness))
                        
                        confidence = min(95.0, 60.0 + (aspect_score + height_score + compactness_score) * 15.0)
                        
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
                            'aspect_ratio': w/h,
                            'height_ratio': h/h_roi,
                            'compactness': compactness
                        })
                        
                        logger.debug(f"Chiffre détecté: {digit} avec confiance {confidence:.1f}%, "
                                   f"ratio h/ROI: {h/h_roi:.2f}, aspect: {w/h:.2f}")
                        
                        # Sauvegarde de l'image de débogage
                        digit_debug_path = os.path.join(DEBUG_FOLDER, 
                                                      f"{debug_timestamp}_digit_{len(digit_info)}_{digit}.jpg")
                        cv2.imwrite(digit_debug_path, digit_roi_resized)
    
    # Tri des chiffres par position horizontale
    digit_info.sort(key=lambda d: d['center_x'])
    
    # ================== FILTRAGE SUPPLÉMENTAIRE ==================
    # Après la détection, filtrer à nouveau pour éliminer les chiffres anormalement hauts
    filtered_by_height = []
    for digit_data in digit_info:
        height_ratio = digit_data['height_ratio']
        # Rejeter les chiffres dont la hauteur dépasse 65% de la ROI
        if height_ratio < 0.65:
            filtered_by_height.append(digit_data)
        else:
            logger.info(f"Ignoré chiffre trop grand: {digit_data['digit']}, "
                       f"hauteur relative: {height_ratio:.2f}")
    
    digit_info = filtered_by_height
    # =====================================================
    
    # Filtrage des doublons
    filtered_digits = []
    if digit_info:
        filtered_digits.append(digit_info[0])
        for i in range(1, len(digit_info)):
            current = digit_info[i]
            previous = filtered_digits[-1]
            
            # Vérifier si les chiffres sont trop proches
            overlap_threshold = previous['w'] * 0.4
            distance = current['center_x'] - previous['center_x']
            
            if distance > overlap_threshold:
                filtered_digits.append(current)
            elif current['confidence'] > previous['confidence'] + 10:
                # Si chevauchement significatif, garder celui avec plus de confiance
                filtered_digits[-1] = current
            # Sinon, ignorer le doublon (déjà gardé le premier)
    
    logger.info(f"Détection terminée: {len(filtered_digits)} chiffre(s) filtrés "
                f"(sur {len(digit_info)} initiaux)")
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
    """Extrait TOUS les chiffres avec détection précise (ROI adaptative)"""
    h, w = image.shape[:2]
    
    logger.info(f"Traitement d'image de taille: {w}x{h}")
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Essayer de détecter automatiquement la région des chiffres
    auto_region = find_digit_region(image)
    
    if auto_region:
        # Utiliser la région détectée automatiquement
        roi_x1, roi_y1, roi_x2, roi_y2 = auto_region
        logger.info(f"Utilisation de la région automatique: {auto_region}")
    else:
        # ROI par défaut (comme dans le code original)
        roi_y1, roi_y2 = int(h * 0.28), int(h * 0.38)
        roi_x1, roi_x2 = int(w * 0.35), int(w * 0.65)
        logger.info(f"Utilisation de la ROI par défaut: {(roi_x1, roi_y1, roi_x2, roi_y2)}")
    
    # Extraction de la ROI
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
    
    # Détection précise de TOUS les chiffres dans la ROI
    digit_info, debug_timestamp = detect_precise_digits(roi, roi_coords, image)
    
    # ================== MODIFICATION IMPORTANTE ==================
    # Déterminer dynamiquement la longueur attendue
    if digit_info:
        # Compter seulement les chiffres (pas 'm³' ou autres)
        detected_digits = [d for d in digit_info if d['digit'].isdigit()]
        expected_length = len(detected_digits) if detected_digits else None
        
        if expected_length and expected_length > 0:
            logger.info(f"Longueur attendue déterminée dynamiquement: {expected_length} chiffres")
        else:
            # Fallback à 6 si aucun chiffre n'est détecté
            expected_length = 6
            logger.info("Aucun chiffre détecté, utilisation du fallback: 6 chiffres")
    else:
        expected_length = 6
        logger.info("Pas de détection de chiffres, utilisation du fallback: 6 chiffres")
    # =====================================================
    
    # Si nous détectons des chiffres
    if digit_info:
        # Trier par position X
        digit_info.sort(key=lambda d: d['center_x'])
        
        # Construction de la lecture à partir des chiffres détectés
        detected_digits = ''.join([d['digit'] for d in digit_info if d['digit'].isdigit()])
        
        # Validation de la lecture avec la longueur attendue dynamique
        reading = validate_reading(detected_digits, expected_length=expected_length)
        
        if reading:
            # Calculer la confiance moyenne
            confidences = [d['confidence'] for d in digit_info if d['digit'].isdigit()]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                confidence = min(99.0, avg_confidence * 0.95)
            else:
                confidence = 70.0
            
            logger.info(f"Détection réussie: {len(digit_info)} chiffres -> {reading} ({confidence:.1f}%)")
        else:
            # Fallback: OCR sur toute la ROI
            config_full = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
            digits_text = pytesseract.image_to_string(roi, config=config_full).replace(" ", "")
            reading = validate_reading(digits_text, expected_length=expected_length)
            
            if reading:
                confidence = 65.0
                logger.info(f"OCR fallback: {reading}")
            else:
                reading = ""
                confidence = 30.0
                logger.warning("Pas assez de chiffres détectés")
    else:
        # Fallback complet avec différentes configurations
        configs = [
            "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789",
            "--psm 8 --oem 3",
            "--psm 6 --oem 3"
        ]
        
        reading = ""
        confidence = 30.0
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(roi, config=config)
                # Rechercher un motif à plusieurs chiffres
                match = re.search(r'(\d{5,})', text)
                if match:
                    reading = validate_reading(match.group(1), expected_length=expected_length)
                    confidence = 60.0
                    logger.info(f"Motif trouvé avec config {config}: {reading}")
                    break
            except Exception as e:
                logger.warning(f"Erreur OCR avec config {config}: {e}")
        
        if not reading:
            # Dernier recours : extraire tous les chiffres
            text = pytesseract.image_to_string(roi, config="--psm 3 --oem 3")
            all_digits = re.findall(r'\d', text)
            if len(all_digits) >= 5:
                reading = validate_reading(''.join(all_digits), expected_length=expected_length)
                confidence = 50.0
                logger.info(f"Extraction depuis texte: {reading}")
            else:
                logger.error("Aucune lecture fiable trouvée")
    
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
    
    # ================== AFFICHAGE AMÉLIORÉ ==================
    # Afficher la lecture complète avec plus d'espace
    if reading:
        # Calculer la largeur du texte
        text = f"Lecture: {reading}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        
        # Arrière-plan pour la lecture (plus grand pour 8 chiffres)
        cv2.rectangle(output_image, 
                     (20, 20),
                     (20 + text_size[0] + 40, 20 + text_size[1] + 30),
                     (0, 0, 0), -1)
        
        # Texte de lecture
        cv2.putText(output_image, text,
                   (30, 20 + text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 255), 3)
        
        # Ajouter aussi le nombre de chiffres détectés
        cv2.putText(output_image, f"({len(digit_info)} chiffres détectés)",
                   (30, 20 + text_size[1] + 40), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (200, 255, 200), 2)
    # =====================================================
    
    # Ajout du timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output_image, f"Traité le: {timestamp}",
               (20, h-20), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)
    
    # Ajouter une légende pour le type de ROI
    roi_type = "Automatique" if auto_region else "Par défaut"
    cv2.putText(output_image, f"ROI: {roi_type}",
               (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)
    
    return reading, confidence, output_image, digit_info, debug_timestamp

# =====================================================
# TRAITEMENT PRINCIPAL
# =====================================================
def process_image(image_path):
    """Fonction de traitement principale avec ROI adaptative"""
    logger.info(f"Début du traitement de l'image: {image_path}")
    
    # Lecture de l'image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Impossible de lire l'image: {image_path}")
        raise ValueError("Impossible de lire l'image")
    
    h, w = image.shape[:2]
    logger.info(f"Image chargée: {w}x{h} pixels")
    
    # Extraction de la lecture avec ROI adaptative
    reading, confidence, output_image, digit_info, debug_timestamp = extract_precise_meter_reading(image)
    
    logger.info(f"Lecture finale: {reading}, Confiance: {confidence:.1f}%")
    
    # Sauvegarde de l'image résultat
    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_path, output_image)
    logger.info(f"Image résultat sauvegardée: {result_path}")
    
    # ================== MODIFICATION IMPORTANTE ==================
    # Préparation des informations détaillées sur les chiffres pour la réponse
    digit_details = []
    reading_digits = list(reading) if reading else []
    
    # Si nous avons des chiffres détectés, les mapper
    if digit_info and reading_digits:
        # Trier les chiffres détectés par position X
        digit_info_sorted = sorted(digit_info, key=lambda d: d['center_x'])
        
        # Mapper chaque chiffre de la lecture aux chiffres détectés
        for i, expected_digit in enumerate(reading_digits):
            detected = False
            digit_confidence = 0.0
            box_info = None
            
            # Essayer de trouver un chiffre détecté à cette position
            if i < len(digit_info_sorted):
                detected_digit = digit_info_sorted[i]
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
    elif reading_digits:
        # Si nous avons une lecture mais pas de détection individuelle
        for i, digit in enumerate(reading_digits):
            digit_details.append({
                'position': i + 1,
                'digit': digit,
                'confidence': f"{confidence * (0.9 - (i * 0.05)):.1f}%",
                'detected': False,
                'status': '⚠',
                'box_coordinates': None
            })
    # =====================================================
    
    # Information sur tous les chiffres détectés
    all_digits_str = ''.join([d['digit'] for d in digit_info]) if digit_info else ""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    result = {
        'reading': reading,
        'confidence': f"{confidence:.1f}%",
        'image_url': f"/uploads/{filename}",
        'debug_url': f"/debug/{debug_timestamp}_original_roi.jpg",
        'digit_details': digit_details,
        'digit_count': len(reading_digits),
        'total_digits_detected': len(digit_info) if digit_info else 0,
        'all_detected_digits': all_digits_str,
        'timestamp': timestamp,
        'debug_timestamp': debug_timestamp
    }
    
    logger.info(f"Traitement terminé. Résultat: {result}")
    return result
    
# =====================================================
# ROUTES FLASK
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

@app.route("/debug_session/<debug_timestamp>")
def debug_session(debug_timestamp):
    """Récupère toutes les images de débogage d'une session"""
    debug_files = []
    
    for file in os.listdir(DEBUG_FOLDER):
        if file.startswith(debug_timestamp):
            file_type = file.replace(debug_timestamp, '').strip('_').split('_')[0]
            debug_files.append({
                'name': file,
                'url': f'/debug/{file}',
                'type': file_type,
                'size': os.path.getsize(os.path.join(DEBUG_FOLDER, file))
            })
    
    return jsonify({
        'session_id': debug_timestamp,
        'debug_files': sorted(debug_files, key=lambda x: x['type'])
    })

@app.route("/upload", methods=["POST"])
def upload():
    try:
        logger.info("Requête d'upload reçue")
        file = request.files.get("image")
        if not file:
            logger.error("Aucune image fournie")
            return jsonify({"error": "Aucune image fournie"}), 400
        
        # Validation du type de fichier
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        if not file.filename.lower().endswith(valid_extensions):
            logger.error(f"Type de fichier invalide: {file.filename}")
            return jsonify({"error": f"Type de fichier invalide. Utilisez {', '.join(valid_extensions)}"}), 400
        
        # Sauvegarde du fichier uploadé
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        input_name = f"input_{timestamp}.jpg"
        input_path = os.path.join(UPLOAD_FOLDER, input_name)
        file.save(input_path)
        logger.info(f"Fichier sauvegardé: {input_path}")
        
        # Traitement de l'image
        result_data = process_image(input_path)
        
        logger.info("Requête traitée avec succès")
        return jsonify({
            "success": True,
            **result_data
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Erreur lors du traitement: {e}\n{error_trace}")
        return jsonify({"error": f"Erreur de traitement: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Water Meter OCR",
        "version": "2.0.0"
    })

@app.route("/debug_images")
def list_debug_images():
    debug_files = []
    for file in os.listdir(DEBUG_FOLDER):
        if file.endswith('.jpg'):
            debug_files.append({
                'name': file,
                'url': f'/debug/{file}',
                'size': os.path.getsize(os.path.join(DEBUG_FOLDER, file)),
                'created': datetime.fromtimestamp(os.path.getctime(os.path.join(DEBUG_FOLDER, file))).isoformat()
            })
    
    return jsonify({
        'debug_files': sorted(debug_files, key=lambda x: x['name'], reverse=True)[:20]
    })

@app.route("/clear_debug", methods=["POST"])
def clear_debug():
    """Efface toutes les images de débogage"""
    try:
        count = 0
        for file in os.listdir(DEBUG_FOLDER):
            if file.endswith('.jpg'):
                os.remove(os.path.join(DEBUG_FOLDER, file))
                count += 1
        
        logger.info(f"{count} fichiers de débogage effacés")
        return jsonify({"success": True, "deleted_count": count})
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")
        return jsonify({"error": str(e)}), 500

# =====================================================
if __name__ == "__main__":
    print("🚰 OCR PRÉCIS POUR COMPTEUR D'EAU - VERSION 2.0")
    print("=============================================")
    print("🌐 Serveur web: http://127.0.0.1:5000")
    print("📁 Dossier d'upload:", os.path.abspath(UPLOAD_FOLDER))
    print("🐛 Dossier de débogage:", os.path.abspath(DEBUG_FOLDER))
    print("📝 Fichier de log:", os.path.abspath("ocr_log.txt"))
    print("🎯 Fonctionnalités améliorées:")
    print("   - ROI adaptative (détection automatique)")
    print("   - Validation intelligente des lectures")
    print("   - Logging détaillé")
    print("   - Support étendu des caractères (inclut m³)")
    print("   - API de débogage améliorée")
    print("   - Interface web enrichie")
    print("=============================================")
    
    # Vérifier que Tesseract est accessible
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR est correctement configuré")
    except Exception as e:
        print(f"❌ Erreur Tesseract: {e}")
        print("⚠️  Assurez-vous que le chemin vers tesseract.exe est correct")
    
    app.run(debug=True, host='0.0.0.0', port=5000)