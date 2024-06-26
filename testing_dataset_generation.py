import os
import cv2 as cv
import mediapipe as mp
import json
from tqdm import tqdm

def process_image(hands, image, label = ""):
    """
    Process an image using the provided Mediapipe Hands object.
    :param hands (mp.solutions.hands.Hands): Mediapipe Hands object for hand landmark detection.
    :param image (str): Path to the image file.
    :param label (str): Label for the image ("like" or "dislike").
    :return all_data (dict): Dictionary containing processed data for the image.
    """
    if label == "dislike":
        label = "dislike"
        label_identifier = 0
    else:
        label = "like"
        label_identifier = 1
        
    try:
        img = cv.imread(image)
        results = hands.process(img[:,:,::-1])
        if results.multi_hand_landmarks:
            all_data = {}
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                my_lm_list = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = lm.x, lm.y
                    my_lm_list.append([px, py])

                # Create a dictionary for the current image
                image_name = image.split("/")[-1]
                print(f"{image_name} is processed successfully")
                current_data = {
                    "label": label,
                    "label_identifier": label_identifier,
                    "landmarks": my_lm_list,
                }
                
                # Add the data to the overall dictionary
                all_data[image_name] = current_data

            return all_data

    except Exception as e:
        print(f"Error processing {image}: {e}")
        return None

def main():
    """
    This function does file processing and passes each image file to the process_image function
    to get the json of each image file and stores it in the keypoints_all.json
    """
    folder_path = ["/local/datasets/csci631/HaGRID/dislike", "/local/datasets/csci631/HaGRID/like"]  
    json_file_path = "keypoints_all.json"

    hands_handler = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    all_data = {}
    for fol_path in folder_path:
        # gets the folder name
        files_type = fol_path.split("/")[-1]
        files = os.listdir(fol_path)
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(fol_path, file)
                data = process_image(hands_handler, image_path, files_type)
                if data:
                    all_data.update(data)

    # Write the overall dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(all_data, json_file, indent=2)

    hands_handler.close()
    print("Finished processing all images.")

if __name__ == "__main__":
    main()