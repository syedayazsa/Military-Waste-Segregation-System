# MILITARY WASTE SEGREGATION USING TRANSFER LEARNING




The accumulation of solid waste in the military is becoming a great concern, and it would result in environmental pollution and may be hazardous to human health if it is not properly managed. It is important to have an advanced/intelligent waste management system to manage a variety of waste materials. One of the most important steps of waste management is the separation of the waste into the different components and this process is normally done manually by hand-picking. To simplify the process, we propose an intelligent waste material classification system, which is developed by using the VGG-16 and VGG-19 Convolutional Neural Network model which is a machine learning tool and serves as the extractor, and dense layers which are used to classify the waste into different groups/types such as boots, guns shells, band-aids, and knives, etc. The proposed system is tested on the trash image dataset which was scraped using a web scraping tool from google images. The separation process of the waste will be faster and intelligent using the proposed waste material classification system without or reducing human involvement.

# Key Issues

  - Modern military operations generate huge waste streams whose mismanagement can have health and environmental consequences.
  - Although burn pits and other practices have been found to present a health risk to personnel, they persist because they are viewed as expedient.
  - Poor waste management practices on bases can lead to air and water pollution that affect communities living in proximity to installations, as well as military personnel and civilian contractors.




# Dataset Information
For this work, a trash image dataset was created by scraping from google images. This is a small dataset and consists of 8106 images, which is divided into eight different classes, pistol/revolver, syringes, knives, bullet shells, bottles, bandaids, boots, automatic rifles.  Some pictures taken randomly from the dataset are shown below.

[![N|Solid](https://i.ibb.co/3smP0Kq/s1.png)](www.google.com)

# Result
Deeper neural networks are more difficult to train due to the necessity of high computation power. Due to this reason, a pre-trained model is used which was trained on ImageNet Dataset. As the number of images in the dataset per class is less and due to limited computational resources, transfer learning has been chosen. Using transfer learning, the model seems to perform well on our validation set after fine-tuning the model. We have used various pre-trained models like VGG-16 and VGG-19 to perform a comparative analysis of those models. The losses and accuracy of each model are analyzed properly. 

This project was performed using Keras library with a TensorFlow backend (version 2.1.4) on the Google Colaboratory platform. During the training of 200 epochs of VGG-16 training, each epoch took about 225 s. and the whole training lasted for one and a half hours. Whereas training VGG-19 took 250s for each epoch and a total of one hour and forty minutes.
 In order to take advantage of model capacity, the model was finely tuned to get the best performance. The learning rate of Adam optimizer was kept as default in Keras model definitions, 0.001 a without weight decay. For the transfer learning experiments the weights of the pre-trained model were initialized to the ImageNet dataset. Simple data augmentation methodologies such as horizontal and vertical flip, zoom, shear, and 15 degrees random rotation were done. For all the training experiments, the batch size was selected as 32. The model was trained on a Tesla K-80 GPU provided by Google Colab for free.

Apart from environmental factors such as lighting, waste could be described as a shape-shifting material. For instance, one can compress a plastic bottle, or tear a paperboard. These materials do not lose their material properties but they lose their key properties to be identified as an intact object. Besides, virtually any object can be an input to a waste sorting system, but the available training samples are limited. This requires the system to generalize extremely well when trained with a relatively small training set. We believe that when the state-of-the-art convolutional neural networks are meticulously trained, they would be capable of producing industrial-grade results to solve these types of problems. The scope of this project was to train a robust CNN model that could generalize well in such scenarios. The accuracies are compared in the table below:

[![N|Solid](https://i.ibb.co/qr3Csv4/s2.png)](www.google.com)

It can be observed that both VGG 19 and VGG 16 have achieved great level of accuracies on the training data and validation data. However, accuracy cannot be relied upon solely for evaluating the performance of the model. One must take into consideration, the validation and training losses to look for overfitting or underfitting. 

[![N|Solid](https://i.ibb.co/hRZwyV9/s3.png)](www.google.com)

From the above losses, a variance can be observed in the validation data even though the model has a decent fit. The performance seems to be good and also does well on new data as shown below:

[![N|Solid](https://i.ibb.co/WtsVyTy/s6.png)](www.google.com)

