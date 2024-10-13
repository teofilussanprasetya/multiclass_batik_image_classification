# Multiclass Batik Image Classification

# Executive Summary

In this project, we aimed to classify images of Indonesian batik motifs into three categories using a multiclass image classification approach. We implemented and compared two deep learning models—VGG16 built from scratch (Model 1) and MobileNetV2 with fewer than 10 million parameters (Model 2). The results show that both models struggled with certain classes, but MobileNetV2 performed slightly better in overall accuracy. However, VGG16 displayed more balanced precision and recall across all classes after tuning.

# Business Problem

Accurate classification of Indonesian batik motifs has significant business implications for industries like fashion, textiles, and cultural preservation. A reliable automated image classification system can help businesses efficiently manage inventories, improve customer experience through better recommendations, and preserve traditional designs. This project focuses on developing an accurate classification system that can handle intricate patterns and variations in batik designs.

# Methodology

- **Data Collection**: We used the ["Indonesian Batik Motifs"](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs) dataset from Kaggle, selecting three classes: batik keraton, batik sekar, and batik sidoluhur, with 147 images in total.
- **Data Preparation**: Images were resized to 256x256 pixels, and data augmentation was applied. The dataset was split into training (80%) and testing (20%) sets.
- **Modeling**:
  - **Model 1 (VGG16)**: Built from scratch with parameter tuning, including batch normalization and optimizer changes.
  - **Model 2 (MobileNetV2)**: Pre-trained with fewer than 10 million parameters.
- **Training**: Both models were trained using early stopping and learning rate reduction with 200 epochs.
- **Evaluation**: We used metrics such as accuracy, precision, recall, F1-score, and confusion matrices to evaluate model performance.

# Skills

- **Deep Learning**: Worked with CNNs and architectures like VGG16 and MobileNetV2.
- **Model Tuning**: Adjusted batch normalization, dropout, and optimizers to improve performance.
- **Evaluation Metrics**: Analyzed precision, recall, F1-score, and accuracy to assess model effectiveness.
- **Python**: Utilized libraries such as TensorFlow and Keras for model development and training.

# Results & Business Recommendation

- **Model 1 (VGG16)** achieved an accuracy of 0.27 in its initial run but improved to 0.37 after tuning. Despite not recognizing classes 0 and 2 initially, class 1 recognition improved during the first tuning attempt, and recognition of class 0 was achieved after the second tuning.
- **Model 2 (MobileNetV2)** had a slightly higher overall accuracy of 0.33. It consistently recognized classes 1 and 2, though it struggled with class 0.

### Visual Results for Model 1 (VGG16):
1. **Sample Prediction Results:**
   ![VGG16 Prediction Results](/Image_batik/vgg16_result.png)
2. **Confusion Matrix:**
   ![VGG16 Confusion Matrix](/Image_batik/vgg16_report.png)
3. **Accuracy Plot:**
   ![VGG16 Accuracy Plot](/Image_batik/vgg16_plot.png)

### Visual Results for Model 2 (MobileNetV2):
1. **Sample Prediction Results:**
   ![MobileNetV2 Prediction Results](
/Image_batik/mobilenetv2_result.png)
2. **Confusion Matrix:**
   ![MobileNetV2 Confusion Matrix](
/Image_batik/mobilenetv2_report.png)
3. **Accuracy Plot:**
   ![MobileNetV2 Accuracy Plot](/Image_batik/mobilenetv2_plot.png)

**Recommendation**: MobileNetV2’s consistency, especially in recognizing classes 1 and 2, makes it a preferable model for deployment. However, further refinement is necessary for both models to improve class 0 recognition. Additionally, transfer learning with a focus on batik-specific feature extraction could improve performance across all classes.

# Next Steps

- **Data Augmentation**: Explore advanced augmentation techniques to better capture batik patterns and improve model generalization.
- **Transfer Learning**: Apply more advanced pre-trained models, such as EfficientNet, to further enhance accuracy.
- **Model Optimization**: Continue tuning hyperparameters such as learning rates, batch sizes, and dropout rates to achieve better balance across all classes.
