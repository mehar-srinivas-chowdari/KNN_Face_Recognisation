import face_recognition_knn as fr



# STEP 1: Train the KNN classifier and save it to disk
# Once the model is trained and saved, you can skip this step next time.
print("Training KNN classifier...")

train_dir_path = "G:/Dataset/Realtime_only_faces"
save_path = "G:/KNN Face_Recognisation/Trained_models/trained_knn_model_12.clf"
classifier = fr.train(train_dir_path, model_save_path=save_path, n_neighbors=4)

print("Training complete!")


