import cv2
import cvlib as cv

stream = cv2.VideoCapture(0)
while True:
    ret, frame = stream.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    bbox, label, conf = cv.detect_common_objects(frame, model='yolov4')
    person_count = label.count('person')
    cv2.putText(frame, f'Persons: {person_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Person Count", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
stream.release()
cv2.destroyAllWindows()
