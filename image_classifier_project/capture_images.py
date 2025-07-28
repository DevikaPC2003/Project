import cv2
import os

# Change these to actual class names or keep Class1, Class2, Class3
classes = ['class 1', 'class 2', 'class 3']
images_per_class = 30  # Change as needed

for label in classes:
    save_path = f'dataset/{label}'
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"📷 Capturing images for {label}. Press 'q' to quit early.")
    
    count = 0
    while count < images_per_class:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(f"Capturing for {label}", frame)
        file_path = os.path.join(save_path, f'{label}_{count}.jpg')
        cv2.imwrite(file_path, frame)
        count += 1
        print(f"✅ Saved {file_path}")
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Finished capturing for {label}\n")
