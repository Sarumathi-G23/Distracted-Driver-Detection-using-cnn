**Project Overview**
This project aims to detect distracted drivers in real-time using a deep learning model. The system classifies driver behaviors into different categories to enhance road safety and reduce accidents caused by inattention or unsafe driving habits.

**Features**
Detects multiple types of driver distractions such as:
Texting while driving
Talking on the phone
Eating or drinking
Reaching behind
Hair or makeup adjustment
Safe driving (focused on the road)
Real-time detection capability
User-friendly interface for monitoring driver behavior
**Dataset**
The model was trained using the State Farm Distracted Driver Dataset (or specify your dataset if different). The dataset consists of labeled images representing different driver behaviors.
**Classes:**
Safe driving
Texting – right hand
Texting – left hand
Talking on the phone – right hand
Talking on the phone – left hand
Operating radio
Drinking
Reaching behind
Hair & makeup
Talking to passenger
**Model**
Architecture: CNN (Convolutional Neural Network) based
Framework: Keras / TensorFlow
Input: Images of the driver (preferably cropped to the driver’s seat)
Output: Predicted class label indicating the driver’s activity
