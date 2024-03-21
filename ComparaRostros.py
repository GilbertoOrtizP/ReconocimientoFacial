import cv2
import face_recognition
import matplotlib.pyplot as plt

image1=cv2.imread("ImagenesRostros/Brad Pitt/Imagen1.jpg")
image1= cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 1)
plt.title("Brad Pitt")
plt.imshow(image1)
#image = cv2.resize(img, (300,450))
image2=cv2.imread("ImagenesRostros/Brad Pitt/Imagen2.jpg")
image2= cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 2)
plt.title("Brad Pitt")
plt.imshow(image2)

face_loc1=face_recognition.face_locations(image1)
vector_rostro1 = face_recognition.face_encodings(image1, known_face_locations=[face_loc1][0])
vector_rostro1=vector_rostro1[0]

face_loc2=face_recognition.face_locations(image2)
vector_rostro2 = face_recognition.face_encodings(image2, known_face_locations=[face_loc2][0])
vector_rostro2=vector_rostro2[0]

resultado = face_recognition.compare_faces([vector_rostro1],vector_rostro2)
print(resultado)