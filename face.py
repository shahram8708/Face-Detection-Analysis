import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            results = DeepFace.analyze(face_roi, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            if isinstance(results, list) and len(results) > 0:
                result = results[0]  
                
                age = result['age']
                gender = result['dominant_gender']
                emotion = result['dominant_emotion']

                print(f"Age: {age}, Gender: {gender}, Emotion: {emotion}")

                cv2.putText(frame, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.putText(frame, f"Gender: {gender}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            else:
                print("No valid results found")
        except Exception as e:
            print(f"Error in analysis: {e}")

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
