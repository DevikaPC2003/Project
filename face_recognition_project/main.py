from deepface import DeepFace
import cv2
import os

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    name = "Unknown"

    try:
        # Save frame as temporary image
        cv2.imwrite("temp.jpg", frame)

        # Perform recognition
        result = DeepFace.find(img_path="temp.jpg", db_path="known_faces", enforce_detection=False)

        if len(result) > 0 and len(result[0]) > 0:
            name = os.path.basename(result[0].iloc[0]['identity']).split(".")[0]
    except Exception as e:
        print("Error:", e)
        name = "Unknown"

    # Display name on frame
    cv2.putText(frame, name, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

