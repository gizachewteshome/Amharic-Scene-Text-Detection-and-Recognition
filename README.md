This repository contains the implementation of a transformer-based approach for Amharic scene text detection and recognition. The project leverages pretraining on synthetic datasets and fine-tuning on domain-specific data to enhance Optical Character Recognition (OCR) performance for the complex Amharic script. It integrates a Vision Transformer (ViT) for recognition and an EAST-like model with a ResNet50 backbone for detection, addressing challenges posed by the intricate syllabic structure of Amharic and limited annotated datasets.

Table of Contents
Overview
Architecture
Scene Text Recognition Module
Scene Text Detection Module
Datasets
Installation
Usage
Training
Evaluation
Results
Future Work
References
Overview
Scene text detection and recognition in Amharic script enable better access to textual information in diverse real-world scenarios. This project proposes a robust pipeline that:

Utilizes transformer-based models with Vision Transformers for effective feature extraction and sequence modeling.
Leverages a combination of synthetic and real-world datasets for pretraining and fine-tuning.
Provides a modular architecture for separate text detection and recognition tasks, tailored specifically for the Amharic script.
The approach demonstrates substantial improvements in recognition accuracy, showing potential for future enhancements in detection performance and handling complex real-world conditions such as distortion and occlusion.

Architecture
Scene Text Recognition Module
Backbone: Vision Transformer (ViT)
The image is divided into patches, processed by multi-head self-attention layers to capture long-range dependencies, and transformed into a sequence of embeddings.

Character-Index Mapping:
A mapping tailored for Amharic characters (including numerals and special symbols) assigns unique indices to each character. Special tokens like PAD, UNK, and CTC blank are handled as needed.

Loss Function:
Utilizes Connectionist Temporal Classification (CTC) loss with log-softmax for training without explicit character-level alignment.

Model Flow:

Image → Patch Embedding → ViT Layers
Sequence Modeling → Recognition Head → CTC Decoding
Scene Text Detection Module
Backbone: ResNet50 integrated with an EAST-like architecture
Features are extracted using ResNet50, followed by a custom decoder to produce bounding boxes for text regions.

Loss Functions:
Combined loss from Dice Loss, Axis-Aligned Bounding Box (AABB) Loss, and Angle Loss to handle text localization, orientation, and region segmentation.

Model Flow:

Input Image → ResNet50 Feature Extraction
Decoder → Multiple Output Heads → Bounding Box Predictions
Datasets
The project utilizes publicly available datasets for training and evaluation:

HUST-AST: Large synthetic dataset for initial pretraining.
HUST-ART: Real-world dataset for fine-tuning detection and recognition.
ABE: Real-world recognition dataset.
Dataset statistics:

Task	Dataset	Training	Testing	Total
Detection	HUST-ART	1,500	700	2,200
HUST-AST	75,904	-	75,904
Recognition	ABE	8,986	3,852	12,838
HUST-ART	7,877	3,377	11,254
HUST-AST	75,904	-	75,904
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/amharic-scene-text-ocr.git
cd amharic-scene-text-ocr
Create and activate a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
Install required packages:

bash
Copy code
pip install -r requirements.txt
Note: The project requires PyTorch, torchvision, and other dependencies specified in requirements.txt.

Usage
Training
The project supports a two-phase training process: pretraining and fine-tuning.

Pretraining Phase:

Train both detection and recognition models on large synthetic datasets.
Example command:
bash
Copy code
python train.py --phase pretrain --task recognition --epochs 5 --lr 0.001 --batch_size 16 --dataset HUST-AST
python train.py --phase pretrain --task detection --epochs 5 --lr 0.001 --batch_size 16 --dataset HUST-AST
Fine-tuning Phase:

Fine-tune models on real-world datasets for improved domain-specific performance.
Example command:
bash
Copy code
python train.py --phase finetune --task recognition --epochs 30 --lr 0.0001 --batch_size 16 --dataset HUST-ART
python train.py --phase finetune --task detection --epochs 20 --lr 0.0001 --batch_size 16 --dataset HUST-ART
Evaluation
After training, evaluate model performance on test sets:

bash
Copy code
python evaluate.py --task recognition --dataset ABE
python evaluate.py --task detection --dataset HUST-ART
Evaluation metrics include precision, recall, and F1-score to measure detection and recognition accuracy.

Results
Recognition Performance:

Achieved an F1-score of 88.12% on HUST-ART test set.
Achieved an F1-score of 84.78% on ABE test set.
Combined dataset F1-score: 86.26%.
Detection Performance:

Detection model shows improvement with extended training, reaching an F1-score of 63.97% on HUST-ART after fine-tuning.
Sample outputs and error analysis demonstrate system strengths and challenges, such as merged bounding boxes or occasional misrecognitions.

