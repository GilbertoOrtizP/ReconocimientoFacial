import cv2
import face_recognition

image = cv2.imread("ImagenesRostros/Brad Pitt/Imagen1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_loc1=face_recognition.face_locations(image)
vector_rostro1 = face_recognition.face_encodings(image, known_face_locations=[face_loc1][0])
vector_rostro1=vector_rostro1[0]

#Video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret,frame = cap.read()
    if ret == False: break
    frame = cv2.flip(frame,1)
    
    face_loc2=face_recognition.face_locations(frame)
    if face_loc2 != []:
        for face_loc2 in face_loc2:
            vector_rostro2 = face_recognition.face_encodings(frame, known_face_locations=[face_loc2])
            vector_rostro2=vector_rostro2[0]
            result = face_recognition.compare_faces([vector_rostro1],vector_rostro2)
            #print("Resultado:",result)
            if result[0]==True:
                text="Brad Pitt"
                color=(125,220,0)
            else:
                text="Desconocido"
                color=(50,50,255)
            
            cv2.rectangle(frame, (face_loc2[3], face_loc2[2]), (face_loc2[1], face_loc2[2]+30),color,-3)
            cv2.rectangle(frame, (face_loc2[3], face_loc2[0]), (face_loc2[1], face_loc2[2]),color,5)
            cv2.putText(frame, text, (face_loc2[3], face_loc2[2]+20),2,0.7,(255,255,255),1)
            
    
    cv2.imshow("Frame",frame)
    k = cv2.waitKey(1)
    if k == 27 & 0xFF:
        break
    
cap.release()
cv2.destroyAllWindows()