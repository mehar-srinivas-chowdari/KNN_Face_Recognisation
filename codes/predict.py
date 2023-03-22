import face_recognition_knn as fr

import os


# STEP 2: Using the trained classifier, make predictions for unknown images
for image_file in os.listdir("G:/KNN Face_Recognisation/Testing_Images/Image_one"):
    full_file_path = os.path.join("G:/KNN Face_Recognisation/Testing_Images/Image_one/", image_file)

    print("Looking for faces in {}".format(image_file))

    # Find all people in the image using a trained classifier model
    # Note: You can pass in either a classifier file name or a classifier model instance
    model_path = 'G:/KNN Face_Recognisation/Trained_models/trained_knn_model_12.clf'
    predictions = fr.predict(full_file_path, model_path = model_path, distance_threshold = 0.48)

    # Print results on the console
    present = []
    for name, (top, right, bottom, left) in predictions:
        #print("- Found {} at ({}, {})".format(name, left, top))
        present.append(name)
    present = list(set(present))
    print("The Students identified in the image {} are".format(image_file))
    for student in present:
        print(student)

    # Display results overlaid on an image
    img_save_path = "G:/KNN Face_Recognisation/output/model12-2/"
    fr.show_prediction_labels_on_image(full_file_path, predictions, img_save_path)
