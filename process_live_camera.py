import cv2 as cv
import mediapipe as mp
import json
from classifier import Classifier
from segmentation import HumanSegmenter

def process_live_camera(hands, model, segmenter):
    """
    Function processes the live camera and prints out the gesture recognised.
    :param hands: hands is the detection module/ hand detection model that detects hand and 
    gives the landmarks.
    :param model: model is the classifier model that is trained on the dataset to classify the gestures.
    """
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        # Flip the frame horizontally for a later selfie-view display
        # frame = cv.flip(frame, 1)

        # Process the frame and get hand landmarks
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                my_lm_list = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = lm.x, lm.y
                    my_lm_list.append([px, py])

                if my_lm_list == []:
                    print("No hand detected")
                    continue
                    
                predicted_label = model.predict_label(my_lm_list )
                print(f"Predicted Label Identifier: {predicted_label}")
                result = segmenter.detect_and_draw(frame, predicted_label)
                # Display the frame
            cv.imshow("Segmented Image", result)
            # cv.imshow("Hand Landmarks", frame)

        else:
            print("hands not detected")
            result = segmenter.detect_and_draw(frame, -1)
            cv.imshow("Segmented Image", result)
                # Display the landmarks on the frame (optional)
                # mp.solutions.drawing_utils.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        # cv.imshow("keypoints", frame)

        if cv.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the live capture
            break

    cap.release()
    cv.destroyAllWindows()

def main():
    hands_handler = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    segmenter = HumanSegmenter()

    with open("keypoints_all.json", 'r') as f:
        data = json.load(f)

    model = Classifier()
    model.train_model(data)

    print("model trained")
    process_live_camera(hands_handler, model, segmenter)

    hands_handler.close()
    print("Finished live capture and processing.")

if __name__ == "__main__":
    main()