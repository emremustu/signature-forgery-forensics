import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import forgery_detector as fd

def load_data_and_extract_features(csv_path="dataset_metadata.csv"):
    print("Loading dataset and extracting features...")
    
    features_b1 = []
    features_b2 = []
    features_prop = []
    labels = []
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row["filepath"]
            label = int(row["label"])
            x_min = int(row["x_min"])
            y_min = int(row["y_min"])
            x_max = int(row["x_max"])
            y_max = int(row["y_max"])
            
            img_path = os.path.join("dataset", filepath)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            crop = fd.crop_signature(img, x_min, y_min, x_max, y_max)
            
            # Extract features
            fb1 = fd.extract_baseline1_edge_shape(crop)
            fb2 = fd.extract_baseline2_lbp(crop)
            fprop = fd.extract_proposed_hybrid(crop)
            
            features_b1.append(fb1)
            features_b2.append(fb2)
            features_prop.append(fprop)
            labels.append(label)
            
    return (np.array(features_b1), 
            np.array(features_b2), 
            np.array(features_prop), 
            np.array(labels))

def generate_visual_samples(csv_path="dataset_metadata.csv"):
    print("Generating visual samples (DFT and LBP plots)...")
    
    # Find one genuine and one forged sample
    gen_row = None
    forg_row = None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["label"]) == 0 and gen_row is None:
                gen_row = row
            if int(row["label"]) == 1 and forg_row is None:
                forg_row = row
            if gen_row and forg_row:
                break
                
    # Load and crop
    img_gen = cv2.imread(os.path.join("dataset", gen_row["filepath"]))
    crop_gen = fd.crop_signature(img_gen, int(gen_row["x_min"]), int(gen_row["y_min"]), int(gen_row["x_max"]), int(gen_row["y_max"]))
    gray_gen = cv2.cvtColor(crop_gen, cv2.COLOR_BGR2GRAY)
    
    img_forg = cv2.imread(os.path.join("dataset", forg_row["filepath"]))
    crop_forg = fd.crop_signature(img_forg, int(forg_row["x_min"]), int(forg_row["y_min"]), int(forg_row["x_max"]), int(forg_row["y_max"]))
    gray_forg = cv2.cvtColor(crop_forg, cv2.COLOR_BGR2GRAY)
    
    # ----------------------------------------------------
    # DFT Magnitude Spectrums Plot
    # ----------------------------------------------------
    f_gen = np.fft.fft2(gray_gen.astype(np.float32))
    fshift_gen = np.fft.fftshift(f_gen)
    mag_gen = np.log(np.abs(fshift_gen) + 1)
    
    f_forg = np.fft.fft2(gray_forg.astype(np.float32))
    fshift_forg = np.fft.fftshift(f_forg)
    mag_forg = np.log(np.abs(fshift_forg) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].imshow(gray_gen, cmap='gray')
    axes[0, 0].set_title("Genuine Signature (Scanned)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mag_gen, cmap='jet')
    axes[0, 1].set_title("Genuine DFT Spectrum")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gray_forg, cmap='gray')
    axes[1, 0].set_title("Forged Signature (Digitally Inserted)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mag_forg, cmap='jet')
    axes[1, 1].set_title("Forged DFT Spectrum (Note Artifacts/Spikes)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("dft_sample.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # LBP Texture Comparison Plot
    # ----------------------------------------------------
    from skimage.feature import local_binary_pattern
    lbp_gen = local_binary_pattern(gray_gen, 8, 1, method="uniform")
    lbp_forg = local_binary_pattern(gray_forg, 8, 1, method="uniform")
    
    hist_gen = fd.extract_baseline2_lbp(crop_gen)
    hist_forg = fd.extract_baseline2_lbp(crop_forg)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes[0, 0].imshow(gray_gen, cmap='gray')
    axes[0, 0].set_title("Genuine Signature")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(lbp_gen, cmap='gray')
    axes[0, 1].set_title("Genuine LBP Image")
    axes[0, 1].axis('off')
    
    axes[0, 2].bar(range(10), hist_gen, color='blue', alpha=0.7)
    axes[0, 2].set_title("Genuine LBP Histogram")
    axes[0, 2].set_ylim(0, 0.5)
    
    axes[1, 0].imshow(gray_forg, cmap='gray')
    axes[1, 0].set_title("Forged Signature")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(lbp_forg, cmap='gray')
    axes[1, 1].set_title("Forged LBP Image")
    axes[1, 1].axis('off')
    
    axes[1, 2].bar(range(10), hist_forg, color='red', alpha=0.7)
    axes[1, 2].set_title("Forged LBP Histogram")
    axes[1, 2].set_ylim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig("lbp_sample.png", dpi=300)
    plt.close()

def main():
    # 1. Generate visual analysis plots
    generate_visual_samples()
    
    # 2. Extract features
    X_b1, X_b2, X_prop, y = load_data_and_extract_features()
    
    # 3. Train-test split (80% train, 20% test)
    X_train_b1, X_test_b1, y_train, y_test = train_test_split(X_b1, y, test_size=0.2, random_state=42)
    X_train_b2, X_test_b2, _, _ = train_test_split(X_b2, y, test_size=0.2, random_state=42)
    X_train_prop, X_test_prop, _, _ = train_test_split(X_prop, y, test_size=0.2, random_state=42)
    
    # 4. Train classifiers
    # Baseline 1: Edge & Shape (using Random Forest Classifier)
    clf_b1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_b1.fit(X_train_b1, y_train)
    y_pred_b1 = clf_b1.predict(X_test_b1)
    y_prob_b1 = clf_b1.predict_proba(X_test_b1)[:, 1]
    
    # Baseline 2: LBP Texture (using SVM Classifier)
    clf_b2 = SVC(probability=True, kernel='rbf', random_state=42)
    clf_b2.fit(X_train_b2, y_train)
    y_pred_b2 = clf_b2.predict(X_test_b2)
    y_prob_b2 = clf_b2.predict_proba(X_test_b2)[:, 1]
    
    # Proposed Hybrid (LBP + Frequency) (using Random Forest Classifier)
    clf_prop = RandomForestClassifier(n_estimators=150, random_state=42)
    clf_prop.fit(X_train_prop, y_train)
    y_pred_prop = clf_prop.predict(X_test_prop)
    y_prob_prop = clf_prop.predict_proba(X_test_prop)[:, 1]
    
    # Proposed Hybrid (LBP + Frequency) with SVM Classifier (for hyperparameter study)
    clf_prop_svm = SVC(probability=True, kernel='rbf', random_state=42)
    clf_prop_svm.fit(X_train_prop, y_train)
    y_pred_prop_svm = clf_prop_svm.predict(X_test_prop)
    y_prob_prop_svm = clf_prop_svm.predict_proba(X_test_prop)[:, 1]
    
    # 5. Evaluate Metrics
    metrics = {}
    for name, y_pred, y_prob in [
        ("Baseline 1 (Edge/Shape RF)", y_pred_b1, y_prob_b1),
        ("Baseline 2 (LBP SVM)", y_pred_b2, y_prob_b2),
        ("Proposed Hybrid (RF)", y_pred_prop, y_prob_prop),
        ("Proposed Hybrid (SVM)", y_pred_prop_svm, y_prob_prop_svm)
    ]:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "AUC": roc_auc,
            "fpr": fpr,
            "tpr": tpr
        }
        
    # 6. Ablation study: evaluating frequency features alone vs hybrid vs texture alone
    # Frequency features alone: proposed features from index 10 onwards
    X_train_freq = X_train_prop[:, 10:]
    X_test_freq = X_test_prop[:, 10:]
    clf_freq = RandomForestClassifier(n_estimators=150, random_state=42)
    clf_freq.fit(X_train_freq, y_train)
    y_pred_freq = clf_freq.predict(X_test_freq)
    y_prob_freq = clf_freq.predict_proba(X_test_freq)[:, 1]
    
    acc_freq = accuracy_score(y_test, y_pred_freq)
    prec_freq = precision_score(y_test, y_pred_freq)
    rec_freq = recall_score(y_test, y_pred_freq)
    f1_freq = f1_score(y_test, y_pred_freq)
    fpr_freq, tpr_freq, _ = roc_curve(y_test, y_prob_freq)
    auc_freq = auc(fpr_freq, tpr_freq)
    
    metrics["Proposed Frequency-Only (RF)"] = {
        "Accuracy": acc_freq,
        "Precision": prec_freq,
        "Recall": rec_freq,
        "F1-Score": f1_freq,
        "AUC": auc_freq,
        "fpr": fpr_freq,
        "tpr": tpr_freq
    }
    
    # Print metrics
    print("\n" + "="*80)
    print("EXPERIMENTAL EVALUATION RESULTS")
    print("="*80)
    print(f"{'Method':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC':<6}")
    print("-"*80)
    for name, m in metrics.items():
        print(f"{name:<30} | {m['Accuracy']:<10.4f} | {m['Precision']:<10.4f} | {m['Recall']:<10.4f} | {m['F1-Score']:<10.4f} | {m['AUC']:<6.4f}")
    print("="*80)
    
    # Save results to CSV
    with open("experimental_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"])
        for name, m in metrics.items():
            writer.writerow([name, m['Accuracy'], m['Precision'], m['Recall'], m['F1-Score'], m['AUC']])
            
    # 7. Plot ROC Curves
    plt.figure(figsize=(8, 6))
    
    # Plot Baseline 1
    fpr_b1, tpr_b1 = metrics["Baseline 1 (Edge/Shape RF)"]["fpr"], metrics["Baseline 1 (Edge/Shape RF)"]["tpr"]
    auc_b1 = metrics["Baseline 1 (Edge/Shape RF)"]["AUC"]
    plt.plot(fpr_b1, tpr_b1, label=f'Baseline 1 (Edge/Shape RF) (AUC = {auc_b1:.4f})', color='green', linestyle='--')
    
    # Plot Baseline 2
    fpr_b2, tpr_b2 = metrics["Baseline 2 (LBP SVM)"]["fpr"], metrics["Baseline 2 (LBP SVM)"]["tpr"]
    auc_b2 = metrics["Baseline 2 (LBP SVM)"]["AUC"]
    plt.plot(fpr_b2, tpr_b2, label=f'Baseline 2 (LBP SVM) (AUC = {auc_b2:.4f})', color='orange', linestyle='-.')
    
    # Plot Proposed Frequency Only
    fpr_fq, tpr_fq = metrics["Proposed Frequency-Only (RF)"]["fpr"], metrics["Proposed Frequency-Only (RF)"]["tpr"]
    auc_fq = metrics["Proposed Frequency-Only (RF)"]["AUC"]
    plt.plot(fpr_fq, tpr_fq, label=f'Proposed Frequency-Only (RF) (AUC = {auc_fq:.4f})', color='purple', linestyle=':')
    
    # Plot Proposed Hybrid (RF)
    fpr_p, tpr_p = metrics["Proposed Hybrid (RF)"]["fpr"], metrics["Proposed Hybrid (RF)"]["tpr"]
    auc_p = metrics["Proposed Hybrid (RF)"]["AUC"]
    plt.plot(fpr_p, tpr_p, label=f'Proposed Hybrid (RF) (AUC = {auc_p:.4f})', color='blue', linewidth=2)
    
    # Plot diagonal reference
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=300)
    plt.close()
    print("Saved ROC curve plot to roc_curves.png")
    
if __name__ == "__main__":
    main()
